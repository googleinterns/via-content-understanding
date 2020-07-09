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
    """Returns a function that encodes captions.

    Arguments:
        language_model: an instance of BaseLanguageModel that is used to encode
            the text.

    Returns: a function that has two parameters: the video id, and the caption
        text. This function then returns a tuple of 3 values. The first value is
        the video id, the second value is the encoded ids, and the third is the
        number of tokens in the encoding.
    """

    def encode_text(text):
        result, tokens = language_model.encode(text.decode("utf-8"))
        return [result], tokens

    def wrapper(video_id, text):
        result, tokens = tf.numpy_function(
            encode_text, [text], (tf.int64, tf.int64))
        result.set_shape(language_model.encoded_shape)

        return video_id, result, tokens

    return wrapper

def get_language_model_inference_function(language_model):
    """Returns a function that inferences with the language model.

    Arguments:
        language_model: an instance of BaseLanguageModel that is used to
            generate contextual embeddings from the text.

    Returns: a function that has three parameters: the video id, the encoded
        ids, and the number of tokens in the tokenized caption. The function
        returns three values, the  video id, the contextual embeddings, and the
        number of tokens in the tokenized caption.
    """

    def inference(ids):
        return language_model(ids)

    def wrapper(video_id, ids, tokens):
        contextual_embeddings = tf.py_function(inference, [ids], tf.float32)
        return video_id, contextual_embeddings, tokens

    return wrapper

def generate_contextal_embeddings(language_model, dataset):
    """Generate the contextual embeddings for a given dataset.

    Arguments:
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

    encode = get_encode_function(language_model)

    return (dataset
        .map(encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(language_model.batch_size)
        .map(get_language_model_inference_function(language_model))
        .unbatch())


def generate_and_cache_contextual_embeddings(language_model, source_dataset):
    """Generate and cache contextual embeddings for a given dataset/model.

    For each split in the source dataset, given by 
    source_dataset.id_caption_pair_dataset, generate contextual embeddings using
    language_model and cache them.

    Arguments:
        language_model: an instance of BaseLanguageModel used for generating
            contextual embeddings.
        source_dataset: the source dataset as an an instance of BaseDataset.

    Returns: nothing.
    """

    for ds_split, split_name in source_dataset.id_caption_pair_datasets:
        ds_split = generate_contextal_embeddings(language_model, ds_split)

        cache_language_model_embeddings(
            ds_split, source_dataset, language_model, split=split_name)
