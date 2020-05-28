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

embeddings_per_file = 1000

decoding_schema = {
    "video_id": tf.io.FixedLenFeature([], tf.string),
    "seralized_embeddings": tf.io.FixedLenFeature([], tf.string)
}


def get_feature(value):
    bytes_list = tf.train.BytesList(value=[value.numpy()])
    return tf.train.Feature(bytes_list=bytes_list)

def get_records_directory(dataset, language_model, split):
    dataset_name = dataset.dataset_name
    language_model_name = language_model.name

    path = Path(f"cached_data/{dataset_name}/{language_model_name}/{split}")

    path.mkdir(parents=True, exist_ok=True)

    return path

def seralize_to_protobuf(video_id, contextual_embeddings):
    video_id_feature = get_feature(video_id)

    seralized_embedding = tf.io.serialize_tensor(contextual_embeddings)
    embeddings_feature = get_feature(seralized_embedding)

    feature = {
        "video_id": video_id_feature,
        "seralized_embeddings": embeddings_feature
    }

    protobuf = tf.train.Example(features=tf.train.Features(feature=feature))

    return protobuf.SerializeToString()

def seralize_to_protobuf_wrapper(*args):
    return tf.py_function(seralize_to_protobuf, args, tf.string)

def write_dataset(dataset, records_directory, dataset_size):
    num_shards = math.ceil(dataset_size / embeddings_per_file)

    for shard_index in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=shard_index)
        file_path = records_directory / (f"lm_{shard_index}.tfrecord")

        writer = tf.data.experimental.TFRecordWriter(str(file_path))
        writer.write(shard)
 
def cache_language_model_embeddings(dataset, source_dataset, language_model,
    split):

    dataset = dataset.unbatch().map(seralize_to_protobuf_wrapper, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    records_directory = get_records_directory(source_dataset, language_model,
        split)

    write_dataset(dataset, records_directory, 
        source_dataset.num_of_examples_by_split(split))

def get_cached_records(source_dataset, language_model, split):

    records_directory = get_records_directory(
        source_dataset, language_model, split)
    glob_string = str((records_directory / "*.tfrecord").absolute())

    return glob.glob(glob_string)

def unseralize_data(seralized_item):
    example = tf.io.parse_single_example(seralized_item, decoding_schema)
    contextual_embeddings = tf.io.parse_tensor(
        example["seralized_embeddings"], tf.float32)

    return (example["video_id"], contextual_embeddings)

def get_cached_language_model_embeddings(source_dataset, language_model, split):
    record_files = get_cached_records(source, split)

    ds = tf.data.TFRecordDataset(record_files)
    ds.map(unseralize_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds
