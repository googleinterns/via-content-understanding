"""Functions for caching tokens and embeddings from language models.

Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
import math
from pathlib import Path
import glob
import random

embeddings_per_file = 250

embeddings_schema = {
    "video_id": tf.io.FixedLenFeature([], tf.string),
    "serialized_embeddings": tf.io.FixedLenFeature([], tf.string),
}

encodings_schema = {
    "video_id": tf.io.FixedLenFeature([], tf.string),
    "serialized_encodings": tf.io.FixedLenFeature([], tf.string),
}

base_path = Path(f"/mnt/disks/fast_ssd/cached_data/")


def get_bytes_feature(value):
    """Gets a tf.train.Feature from the parameter value."""
    bytes_list = tf.train.BytesList(value=[value.numpy()])
    return tf.train.Feature(bytes_list=bytes_list)

def get_records_directory(dataset, language_model, split, postfix=""):
    """Gets a path to cache the data from the dataset/model/split."""
    dataset_name = dataset.dataset_name
    language_model_name = language_model.name

    path = base_path / f"{dataset_name}/{language_model_name}/{split}{postfix}"

    path.mkdir(parents=True, exist_ok=True)

    return path

def serialize_to_protobuf(video_id, contextual_embeddings, tokens):
    """Serializes the video_id and contextual_embeddings.

    Parameters:
        video_id: the id of the video corresponding to the caption.
        contextual_embeddings: a 1 x padded size x embedding dimension tensor.
        tokens: the number of tokens created from the original caption.

    Returns:
        A protobuf serialized as a string.
    """
    video_id_feature = get_bytes_feature(video_id)

    # Accessing contextual_embeddings[0] to gets rid of the extra dimension.
    serialized_embedding = tf.io.serialize_tensor(
        contextual_embeddings)
    embeddings_feature = get_bytes_feature(serialized_embedding)

    feature = {
        "video_id": video_id_feature,
        "serialized_embeddings": embeddings_feature,
    }

    protobuf = tf.train.Example(features=tf.train.Features(feature=feature))

    serialized_protobuf = protobuf.SerializeToString()

    return serialized_protobuf

def serialize_to_protobuf_wrapper(*args):
    """Wraps the serialize_to_protobuf function with tf.py_function."""
    return tf.py_function(serialize_to_protobuf, args, tf.string)

def serialize_encodings(video_id, encodings, tokens):
    serialized_encodings = tf.io.serialize_tensor(encodings)

    video_id_feature = get_bytes_feature(video_id)
    encodings_feature = get_bytes_feature(serialized_encodings)

    feature = {
        "video_id": video_id_feature,
        "serialized_encodings": encodings_feature,
    }

    protobuf = tf.train.Example(features=tf.train.Features(feature=feature))

    serialized_protobuf = protobuf.SerializeToString()

    return serialized_protobuf

def serialize_encodings_wrapper(*args):
    """Wraps the serialize_encodings function with tf.py_function."""
    return tf.py_function(serialize_encodings, args, tf.string)

def write_dataset(dataset, records_directory):
    """Shards a tf.data Dataset and writes it to disk."""
    dataset = dataset.batch(embeddings_per_file).prefetch(
        tf.data.experimental.AUTOTUNE)

    for shard_index, batch in enumerate(dataset):
        file_path = records_directory / (f"lm_{shard_index}.tfrecord")

        shard = tf.data.Dataset.from_tensor_slices(batch)

        # Utilize tf.data.experimental.TFRecordWriter to write large numbers of
        # serialized protocol buffers to TFRecords.
        writer = tf.data.experimental.TFRecordWriter(str(file_path))
        writer.write(shard)
 
def cache_language_model_encodings(dataset, source_dataset, language_model,
    split):
    dataset = dataset.map(
        serialize_encodings_wrapper,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    records_directory = get_records_directory(
        source_dataset, language_model, split, "encodings")

    write_dataset(dataset, records_directory)

def cache_language_model_embeddings(dataset, source_dataset, language_model,
    split):
    """Caches embeddings for a specific dataset/model/split.

    Parameters:
        dataset: a tf.data Dataset to be cached. Each element of the dataset
            should be a tuple where the first element is the video id as a
            string tensor, the second is the contextual embeddings as a float32
            tensor, and the third is the number of tokens in the tokenized raw
            caption as an integer tensor.
        source_dataset: an instance of BaseDataset for the dataset to be cached.
        language_model: an instance of BaseLanguageModel for the language model
            used to generate the contextual embeddings.
        split: the name of the split (as a string).
    """

    dataset = dataset.map(
        serialize_to_protobuf_wrapper,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    records_directory = get_records_directory(
        source_dataset, language_model, split)

    write_dataset(dataset, records_directory)

def get_cached_records_dataset(
    source_dataset, language_model, split, shuffle_files, postfix=""):
    """Gets a TFRecordDataset of cached data.

    Parameters:
        source_dataset: an instance of BaseDataset that the cached embeddings
            were generated from. 
        language_model: an instance of BaseLanguage model that the cached
            embeddings were generated from.
        split: the name of the dataset split.
        shuffle_files: a boolean indicating if the order in which the files are
            read should be random or not.

    Returns: a TFRecord dataset of serialized embeddings.
    """

    records_directory = get_records_directory(
        source_dataset, language_model, split, postfix)
    glob_string = str((records_directory / "*.tfrecord").absolute())

    file_paths = glob.glob(glob_string)

    if len(file_paths) == 0:
        # In this case, there are no files, so an empty dataset is returned.
        return tf.data.TFRecordDataset(file_paths)

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    if shuffle_files:
        dataset = dataset.shuffle(len(file_paths))

    dataset = dataset.batch(len(file_paths)).interleave(
        lambda files: tf.data.TFRecordDataset(files))

    return dataset


def unserialize_embeddings_wrapper(text_max_length):
    """Wrapper for unserialize function.

    Parameters:
        text_max_length: the length to zero-pad the contextual embeddings to.
        contextual_embeddings_dim: the last dimension of the contextual
            embeddings.

    Returns: a function that maps from a serialized protobuf string to a tuple
        with two elements: the first being the video id, the second being a
        tensor of size text_max_length x contextual_embeddings_dim.
    """ 

    def get_embedding_length(embedding):
        return embedding.shape[0]

    def unserialize_data(serialized_item):
        """Unserializes a serialized protobuf feature.

        Returns: a tuple of 2 items, the first being the video id as a string
            tensor, the second being the contextual embeddings.
        """
        example = tf.io.parse_single_example(serialized_item, embeddings_schema)
        video_id = example["video_id"]

        contextual_embeddings = tf.io.parse_tensor(
            example["serialized_embeddings"], tf.float32)

        return (video_id, contextual_embeddings)
        embedding_length = tf.numpy_function(
            get_embedding_length, [contextual_embeddings], tf.int64)

        if embedding_length >= text_max_length:
            return (video_id, contextual_embeddings[:text_max_length])
        else:
            output = tf.zeros((
                text_max_length - embedding_length,
                contextual_embeddings_dim))
            
            output = tf.concat([contextual_embeddings, output], axis=0)
            return (video_id, output)

    return unserialize_data

def unserialize_encodings_wrapper(text_max_length):
    """Wrapper for unserialize function.

    Parameters:
        text_max_length: the length to zero-pad the contextual embeddings to.
        contextual_embeddings_dim: the last dimension of the contextual
            embeddings.

    Returns: a function that maps from a serialized protobuf string to a tuple
        with two elements: the first being the video id, the second being a
        tensor of size text_max_length x contextual_embeddings_dim.
    """ 

    def get_encoding_length(encoding):
        return encoding.shape[0]

    def unserialize_data(serialized_item):
        """Unserializes a serialized protobuf feature.

        Returns: a tuple of 2 items, the first being the video id as a string
            tensor, the second being the contextual embeddings.
        """
        example = tf.io.parse_single_example(serialized_item, encodings_schema)
        video_id = example["video_id"]

        encodings = tf.io.parse_tensor(
            example["serialized_encodings"], tf.int64)

        encoding_length = tf.numpy_function(
            get_encoding_length, [encodings], tf.int64)

        return (video_id, encodings, encoding_length)

        if encoding_length >= text_max_length:
            return (video_id, encodings[:text_max_length], encoding_length)
        else:
            output = tf.zeros(text_max_length - encoding_length, tf.int64)
            
            output = tf.concat([encodings, output], axis=0)
            return (video_id, output, encoding_length)

    return unserialize_data

def get_cached_language_model_encodings(
    source_dataset, language_model, split, shuffle_files=True):
    dataset = get_cached_records_dataset(
        source_dataset, language_model, split, shuffle_files, "encodings")

    return dataset.map(
        unserialize_encodings_wrapper(
            language_model.contextual_embeddings_shape[0]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_cached_language_model_embeddings(
    source_dataset, language_model, split, shuffle_files=True):
    """Loads the cached embeddings for a specific dataset/split.

    Parameters:
        source_dataset: an instance of BaseDataset for the dataset to be loaded.
        language_model: an instance of BaseLanguageModel for the language model
            used to generate the contextual embeddings.
        split: the name of the split (as a string).
        shuffle_files: a boolean indicating if the files should be shuffled
            before they are read.
    
    Returns: a tf.data Dataset where the first element is the video id as a
        string tensor, the second is the contextual embeddings as a float32
        tensor. The dataset is empty if there are no cached results.  
    """
    dataset = get_cached_records_dataset(
        source_dataset, language_model, split, shuffle_files)

    return dataset.map(
        unserialize_embeddings_wrapper(
            *language_model.contextual_embeddings_shape),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
