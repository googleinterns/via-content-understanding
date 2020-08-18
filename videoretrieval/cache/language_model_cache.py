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

embeddings_per_file = 300

embeddings_schema = {
    "video_id": tf.io.FixedLenFeature([], tf.string),
    "serialized_embeddings": tf.io.FixedLenFeature([], tf.string),
}

encodings_schema = {
    "video_id": tf.io.FixedLenFeature([], tf.string),
    "serialized_encodings": tf.io.FixedLenFeature([], tf.string),
    "serialized_attention_masks": tf.io.FixedLenFeature([], tf.string)
}

base_path = Path("./cached_data/")

def get_bytes_feature(value):
    """Gets a tf.train.Feature from the parameter value."""
    bytes_list = tf.train.BytesList(value=[value.numpy()])
    return tf.train.Feature(bytes_list=bytes_list)

def get_records_directory(dataset, language_model, split, type_=""):
    """Gets a path to cache the data from the dataset/model/split.

    Args:
        dataset: the dataset the directory is for. This object should extend
            BaseDataset and have the attribute dataset_name.
        language_model: the language model the directory is for. Should extend
            BaseLanguageModel and have the attribute name.
        split: a string that names the split of the dataset (train, valid, test)
        type_: a string that is appended after the split name to specify the
            type of data stored. Defaults to a blank string.
    """
    dataset_name = dataset.dataset_name
    language_model_name = language_model.name

    path = base_path / f"{dataset_name}/{language_model_name}/{split}{type_}"
    path.mkdir(parents=True, exist_ok=True)

    return path

def serialize_to_protobuf(video_id, contextual_embeddings, attention_mask):
    """Serializes the video_id and contextual_embeddings.

    Parameters:
        video_id: the id of the video corresponding to the caption.
        contextual_embeddings: a padded size x embedding dimension tensor.
        attention_mask: the attention mask for the encodings for the original
            caption.

    Returns:
        A protobuf serialized as a string.
    """
    tokens = tf.concat(
        [
            tf.where(attention_mask == 0),
            tf.constant(
                [[attention_mask.shape[0]]],
                dtype=tf.int64)
        ],
        axis=0)[0][0]

    # Don't store the zero-ed out parts of the embeddings
    serialized_embedding = tf.io.serialize_tensor(
        contextual_embeddings[:tokens])

    video_id_feature = get_bytes_feature(video_id)
    embeddings_feature = get_bytes_feature(serialized_embedding)

    feature = {
        "video_id": video_id_feature,
        "serialized_embeddings": embeddings_feature,
    }

    protobuf = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized_protobuf = protobuf.SerializeToString()
    return serialized_protobuf

def serialize_to_protobuf_wrapper(*args):
    def serialize_embeddings(caption_info):
        return (
            tf.py_function(serialize_to_protobuf, caption_info, tf.string),
            caption_info[1],
            caption_info[2])
    return tf.map_fn(serialize_embeddings, args)[0]

def serialize_encodings(video_id, encodings, attention_mask):
    """Serializes a video id, encodings, and an attention mask to a protobuf.

    Args:
        video_id: the id of the video this data corresponds to.
        encodings: the encodings that should be stored.
        attention_mask: the attention mask for the encodings.

    Returns:
        A protobuf serialized as a string.
    """ 
    serialized_video_ids = tf.io.serialize_tensor(video_id)
    serialized_encodings = tf.io.serialize_tensor(encodings)
    serialized_attention_masks = tf.io.serialize_tensor(attention_mask)

    video_id_feature = get_bytes_feature(serialized_video_ids)
    encodings_feature = get_bytes_feature(serialized_encodings)
    attention_masks_feature = get_bytes_feature(serialized_attention_masks)

    feature = {
        "video_id": video_id_feature,
        "serialized_encodings": encodings_feature,
        "serialized_attention_masks": attention_masks_feature,
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
 
def cache_language_model_encodings(
    dataset, source_dataset, language_model, split):
    """Caches text encodings for a specific dataset/model/split.

    Args:
        dataset: a tf.data Dataset to be cached. Each element of the dataset
            should be a tuple where the first element is the video id as a
            string tensor, the second is the encodings as an int64 tensor, and
            the third is the attention mask for the given encoding.
        source_dataset: an instance of BaseDataset for the dataset to be cached.
        language_model: an instance of BaseLanguageModel for the language model
            used that the encodings will be inputted to.
        split: the name of the split (as a string).
    """
    dataset = dataset.map(
        serialize_encodings_wrapper,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    records_directory = get_records_directory(
        source_dataset, language_model, split, type_="encodings")
    write_dataset(dataset, records_directory)

def cache_language_model_embeddings(
    dataset, source_dataset, language_model, split):
    """Caches embeddings for a specific dataset/model/split.

    Args:
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
    dataset = (dataset
        .map(
            serialize_to_protobuf_wrapper,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .unbatch())
    records_directory = get_records_directory(
        source_dataset, language_model, split)
    write_dataset(dataset, records_directory)

def get_cached_records_dataset(
    source_dataset, language_model, split, shuffle_files, type_=""):
    """Gets a TFRecordDataset of cached data.

    Args:
        source_dataset: an instance of BaseDataset that the cached embeddings
            were generated from. 
        language_model: an instance of BaseLanguage model that the cached
            embeddings were generated from.
        split: the name of the dataset split.
        shuffle_files: a boolean indicating if the order in which the files are
            read should be random or not.
        type_: a string that is appended after the split name to specify the
            type of data stored. Defaults to a blank string.

    Returns: a TFRecord dataset of serialized embeddings.
    """

    records_directory = get_records_directory(
        source_dataset, language_model, split, type_)
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

    Args:
        text_max_length: the length to zero-pad the contextual embeddings to.
        contextual_embeddings_dim: the last dimension of the contextual
            embeddings.

    Returns: a function that maps from a serialized protobuf string to a tuple
        with two elements: the first being the video id, the second being a
        tensor of size text_max_length x contextual_embeddings_dim.
    """ 

    def get_embedding_length(embedding):
        return embedding.shape[0]

    def get_embedding_dim(embedding):
        return embedding.shape[-1]

    def unserialize_data(serialized_item):
        """Unserializes a serialized protobuf feature.

        Returns: a tuple of 2 items, the first being the video id as a string
            tensor, the second being the contextual embeddings.
        """
        example = tf.io.parse_single_example(serialized_item, embeddings_schema)
        video_id = example["video_id"]

        contextual_embeddings = tf.io.parse_tensor(
            example["serialized_embeddings"], tf.float32)
        embedding_length = tf.py_function(
            get_embedding_length, [contextual_embeddings], tf.int64)
        embedding_dim = tf.py_function(
            get_embedding_dim, [contextual_embeddings], tf.int64)
        extra_padding_tokens_needed = tf.math.maximum(
            text_max_length - embedding_length, tf.constant(0, tf.int64))
        contextual_embeddings = tf.concat(
            [contextual_embeddings, tf.zeros((
                extra_padding_tokens_needed, embedding_dim), dtype=tf.float32)],
            axis=0)
        return (video_id, contextual_embeddings)

    return unserialize_data

def unserialize_encodings(serialized_item):
    """Unserializes a serialized encodings protocol buffer.

    Args:
        serialized_item: the protocol buffer containing the encodings serialized
            as a string.

    Returns: a tuple with three elements: the first being the video id, the
        second being the encoded captions, and third being the attention mask.
    """ 

    example = tf.io.parse_single_example(serialized_item, encodings_schema)
    video_id = tf.io.parse_tensor(example["video_id"], tf.string)

    encodings = tf.io.parse_tensor(
        example["serialized_encodings"], tf.int64)
    attention_mask = tf.io.parse_tensor(
        example["serialized_attention_masks"], tf.int64)
    return (video_id, encodings, attention_mask)

def get_cached_language_model_encodings(
    source_dataset, language_model, split, shuffle_files=True):
    """Loads cached encodings for a specific dataset/split.

    Args:
        source_dataset: an instance of BaseDataset for the dataset to be loaded.
        language_model: an instance of BaseLanguageModel for the language model
            used to generate the contextual embeddings.
        split: the name of the split (as a string).
        shuffle_files: a boolean indicating if the files should be shuffled
            before they are read.
    
    Returns: a tf.data Dataset where each item is a tuple of three items,
        ordered such that captions for the same video will be adjacent to each
        other. In this tuple, the first item is the video id as a string tensor.
        The second is caption text encoded as an int64 tensor for a language
        model, and the third is the attention mask as an int64 tensor for the
        language model.
    """
    dataset = get_cached_records_dataset(
        source_dataset, language_model, split, shuffle_files, type_="encodings")

    return dataset.map(
        unserialize_encodings,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch()


def get_cached_language_model_embeddings(
    source_dataset, language_model, split, shuffle_files=True):
    """Loads the cached embeddings for a specific dataset/split.

    Args:
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
            language_model.contextual_embeddings_shape[0]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
