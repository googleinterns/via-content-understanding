"""Copyright 2020 Google LLC

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

embeddings_per_file = 100

decoding_schema = {
    "video_id": tf.io.FixedLenFeature([], tf.string),
    "seralized_embeddings": tf.io.FixedLenFeature([], tf.string),
    "text": tf.io.FixedLenFeature([], tf.string),
}

base_path = Path(f"/mnt/disks/fast_ssd/cached_data/")


def get_feature(value):
    """Gets a tf.train.Feature from the parameter value as a bytes list."""
    bytes_list = tf.train.BytesList(value=[value.numpy()])
    return tf.train.Feature(bytes_list=bytes_list)

def get_records_directory(dataset, language_model, split):
    """Gets a path to cache the data from the dataset/model/split."""
    dataset_name = dataset.dataset_name
    language_model_name = language_model.name

    path = base_path / f"{dataset_name}/{language_model_name}/{split}"

    path.mkdir(parents=True, exist_ok=True)

    return path

def seralize_to_protobuf(video_id, contextual_embeddings, text):
    """Seralizes the video_id, contextual_embeddings, and text as a protobuf."""

    video_id_feature = get_feature(video_id)
    text_feature = get_feature(text)

    seralized_embedding = tf.io.serialize_tensor(contextual_embeddings)
    embeddings_feature = get_feature(seralized_embedding)

    feature = {
        "video_id": video_id_feature,
        "seralized_embeddings": embeddings_feature,
        "text": text_feature
    }

    protobuf = tf.train.Example(features=tf.train.Features(feature=feature))

    seralized = protobuf.SerializeToString()

    return seralized

def seralize_to_protobuf_wrapper(*args):
    """Wraps the seralize_to_protobuf function with tf.py_function."""
    return tf.py_function(seralize_to_protobuf, args, tf.string)

def write_dataset(dataset, records_directory, dataset_size):
    """Shards a tf.data Dataset and writes it to disk."""
    dataset = dataset.batch(embeddings_per_file).prefetch(
        tf.data.experimental.AUTOTUNE)

    for shard_index, batch in enumerate(dataset):
        file_path = records_directory / (f"lm_{shard_index}.tfrecord")

        shard = tf.data.Dataset.from_tensor_slices(batch)

        writer = tf.data.experimental.TFRecordWriter(str(file_path))
        writer.write(shard)
 
def cache_language_model_embeddings(dataset, source_dataset, language_model,
    split):
    """Caches embeddings for a specific dataset/model/split.

    Arguments:
        dataset: a tf.data Dataset to be cached where the first element is the
            video is as a string tensor, the second is the contextual embeddings
            as a float32 tensor, and the third is the raw caption text as a
            string tensor.
        source_dataset: an implementation of BaseDataset for the dataset to be
            loaded.
        language_model: an implementation of BaseLanguageModel for the language
            model used to generate the contextual embeddings.
        split: the name of the split (as a string).

    Returns: a tf.data Dataset where the first element is the video id as a
        string tensor, the second is the contextual embeddings as a float32
        tensor, and the third is the raw caption text as a string tensor.
        The dataset is empty if there are no cached results.  
    """

    dataset = dataset.unbatch().map(seralize_to_protobuf_wrapper, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    records_directory = get_records_directory(source_dataset, language_model,
        split)

    write_dataset(dataset, records_directory, 
        source_dataset.num_of_examples_by_split(split))

def get_cached_records_dataset(
    source_dataset, language_model, split, shuffle_files):
    """Gets a TFRecordDataset of cached data.

    Parameters:
        source_dataset: an implementation of BaseDataset that the cached
            embeddings were generated from. 
        language_model: an implementation of BaseLanguage model that the
            cached embeddings were generated from.
        split: the name of the dataset split.
        shuffle_files: a boolean indicating if the files should be shuffled
            before they are read.

    Returns: a list of file paths (as strings) to each cached tfrecord file.
    """

    records_directory = get_records_directory(
        source_dataset, language_model, split)
    glob_string = str((records_directory / "*.tfrecord").absolute())

    file_paths = glob.glob(glob_string)

    if len(file_paths) == 0:
        # In this case, there are no files, so we'll return an empty dataset.
        return tf.data.TFRecordDataset(file_paths)

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    if shuffle_files:
        dataset = dataset.shuffle(len(file_paths))

    dataset = dataset.batch(len(file_paths)).interleave(
        lambda files: tf.data.TFRecordDataset(
            files))#, num_parallel_reads=tf.data.experimental.AUTOTUNE))

    return dataset

def unseralize_data(seralized_item):
    """Unseralizes a seralized protobuf feature.

    Returns: a tuple of 3 items, the first being the video id as a string
        tensor, the second being the contextual embeddings as a float32 tensor,
        the third being the source text as tensor.
    """
    example = tf.io.parse_single_example(seralized_item, decoding_schema)
    contextual_embeddings = tf.io.parse_tensor(
        example["seralized_embeddings"], tf.float32)

    return (example["video_id"], contextual_embeddings, example["text"])

def truncate_contextual_embeddings_wrapper(
    text_max_length, contextual_embeddings_dim):
    """Generates a function that returns."""
    
    def truncate_contextual_embeddings(video_id, contextual_embeddings, text):
        num_words = len(text.decode("utf-8").split(" "))
        if num_words >= text_max_length:
            return video_id, contextual_embeddings[:text_max_length]

        output = tf.zeros(
            (text_max_length - num_words, contextual_embeddings_dim),
            tf.float32)
        output = tf.concat([contextual_embeddings[:num_words], output], axis=0)
        
        return video_id, output

    return lambda *args: tf.numpy_function(
        truncate_contextual_embeddings, args, (tf.string, tf.float32))

def get_cached_language_model_embeddings(
    source_dataset, language_model, split, shuffle_files=True):
    """Loads the cached embeddings for a specific dataset/model/split.

    Arguments:
        source_dataset: an implementation of BaseDataset for the dataset to be
            loaded.
        language_model: an implementation of BaseLanguageModel for the language
            model used to generate the contextual embeddings.
        split: the name of the split (as a string).
        shuffle_files: a boolean indicating if the files should be shuffled
            before they are read.
    Returns: a tf.data Dataset where the first element is the video id as a
        string tensor, the second is the contextual embeddings as a float32
        tensor, and the third is the raw caption text as a string tensor.
        The dataset is empty if there are no cached results.  
    """
    dataset = get_cached_records_dataset(
        source_dataset, language_model, split, shuffle_files)

    return (dataset
        .map(unseralize_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(
            truncate_contextual_embeddings_wrapper(
                *language_model.contextual_embeddings_shape),
            num_parallel_calls=tf.data.experimental.AUTOTUNE))
