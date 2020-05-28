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

Functions for inferencing with a language model.
"""
import tensorflow as tf
from cache import cache_language_model_embeddings 


def get_encode_function(language_model):
    def encode_text(text):
        result = language_model.encode(str(text.numpy()))
        return [result]

    def wrapper(video_id, text):
        result = tf.py_function(encode_text, [text], tf.int64)
        result.set_shape(language_model.encoded_shape)

        return video_id, result

    return wrapper

def get_language_model_inference_function(language_model):
    def wrapper(video_id, ids):
        return video_id, language_model(ids)

    return wrapper

def generate_contextal_embeddings(language_model, dataset):
    encode = get_encode_function(language_model)

    return (dataset
        .map(encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(language_model.batch_size)
        .map(get_language_model_inference_function(language_model)))


def generate_and_cache_contextual_embeddings(language_model, source_dataset):
    for ds_split, split_name in source_dataset.id_caption_pair_datasets:
        ds_split = generate_contextal_embeddings(language_model, ds_split)

        cache_language_model_embeddings(
            ds_split, source_dataset, language_model, split=split_name)
