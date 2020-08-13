"""Functions for inferencing with a language model.

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
from cache import cache_language_model_embeddings
from cache import cache_language_model_encodings
import numpy as np


def get_encode_function(language_model, captions_per_video):
    """Wraps a function that uses a language model's tokenizer to encode text.

    Args:
        language_model: an instance of BaseLanguageModel that is used to encode
            the text.

    Returns: a function that has two parameters: the video id and the caption
        text. This function then returns a tuple of 3 values. The first value is
        the video id, the second value is the encoded ids, and the third is the
        number of tokens in the encoding.
    """

    def encode_text(text):
        text = list(np.char.decode(text.astype(np.bytes0), encoding="utf-8"))
        result, attention_mask = language_model.encode(text)
        return result, attention_mask

    def wrapper(video_id, text):
        result, attention_mask = tf.numpy_function(
            encode_text, [text], (tf.int64, tf.int64))
        result.set_shape(
            (captions_per_video, language_model.encoded_shape[0]))
        attention_mask.set_shape(
            (captions_per_video, language_model.encoded_shape[0]))

        return video_id, result, attention_mask

    return wrapper

def get_language_model_inference_function(language_model):
    """Wraps a function that uses a language model to generate embeddings.

    Args:
        language_model: an instance of BaseLanguageModel that is used to
            generate contextual embeddings from the text.

    Returns: a function that has three parameters: the video id, the encoded
        ids, and the number of tokens in the tokenized caption. The function
        returns three values, the video id, the contextual embeddings, and the
        number of tokens in the tokenized caption.
    """

    def inference(ids, attention_mask):
        return language_model(ids, attention_mask)

    def wrapper(video_id, ids, attention_mask):
        contextual_embeddings = tf.py_function(
            inference, [ids, attention_mask], tf.float32)

        contextual_embeddings = contextual_embeddings * tf.cast(
            attention_mask, tf.float32)[:, :, None]

        return video_id, contextual_embeddings, attention_mask

    return wrapper

def generate_contextual_embeddings(language_model, dataset, captions_per_video):
    """Generate the contextual embeddings for a given dataset.

    Args:
        language_model: an instance of BaseLanguageModel that is used to
            generate contextual embeddings.
        dataset: a tf.data Dataset where each of the dataset has two items.
            First, a video id in the form of a string tensor, and second, 
            a caption corresponding to that video in the form of a string
            tensor.

    Returns: a tf.data Dataset that has three elements: a video id in the form
        of a string tensor, the contextual embeddings as a float32 tensor, and
        the number of tokens in the embedding.
    """

    encode = get_encode_function(language_model, captions_per_video)

    return (dataset
        .batch(captions_per_video)
        .map(encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(get_language_model_inference_function(language_model)))


def generate_and_cache_contextual_embeddings(language_model, source_dataset):
    """Generate and cache contextual embeddings for a given dataset/model.

    For each split in the source dataset, given by 
    source_dataset.id_caption_pair_dataset, generate contextual embeddings using
    language_model and cache them.

    Args:
        language_model: an instance of BaseLanguageModel used for generating
            contextual embeddings.
        source_dataset: the source dataset as an an instance of BaseDataset.
    """

    for ds_split, split_name in source_dataset.id_caption_pair_datasets:
        ds_split = generate_contextual_embeddings(
            language_model, ds_split, source_dataset.captions_per_video)

        cache_language_model_embeddings(
            ds_split, source_dataset, language_model, split=split_name)

def generate_and_cache_encodings(language_model, source_dataset):
    """Generates and caches encodings for a given dataset/language model.

    For each split in the source dataset, given by
    source_dataset.id_caption_pair_dataset, generate text encodings for
    language_model and cache them.

    Args:
        language_model: an instance of BaseLanguageModel used for generating
            encodings.
        source_dataset: the source dataset as an an instance of BaseDataset.
    """
    for ds_split, split_name in source_dataset.id_caption_pair_datasets:
        generate_encodings = get_encode_function(
            language_model, source_dataset.captions_per_video)

        ds_split = (ds_split
            .batch(source_dataset.captions_per_video)
            .map(
                generate_encodings,
                num_parallel_calls=tf.data.experimental.AUTOTUNE))

        cache_language_model_encodings(
            ds_split, source_dataset, language_model, split=split_name)

