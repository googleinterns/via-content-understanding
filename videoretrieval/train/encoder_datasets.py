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

Wrappers for datasets for encoders.

"""

import cache
import tensorflow as tf
import numpy as np

def replace_video_id_with_expert_features_wrapper(precomputed_features):
    output_shape = len(precomputed_features) * (tf.float32,)

    def get_expert_features(video_id_encoded):
        expert_features = []

        video_id = video_id_encoded.decode("utf-8")

        for feature_dict in precomputed_features:
            expert_features.append(feature_dict[video_id].astype(np.float32))

        return expert_features

    def wrapper(video_id, ids):
        expert_features = tf.numpy_function(
            get_expert_features, [video_id], output_shape)

        return (video_id, tuple(expert_features), ids)

    return wrapper

def update_dataset_shape_wrapper(experts, language_model):
    expert_shapes = [expert.embedding_shape for expert in experts]
    contextual_embeddings_shape = language_model.contextual_embeddings_shape

    def map_fn(video_id, expert_features, contextual_embeddings):
        for expert_feature, shape in zip(expert_features, expert_shapes):
            expert_feature.set_shape(shape)

        contextual_embeddings.set_shape(contextual_embeddings_shape)

        return video_id, expert_features, contextual_embeddings

    return map_fn


def match_cached_embeddings_with_experts(
    language_model, experts, precomputed_features, *datasets):
    map_fn = replace_video_id_with_expert_features_wrapper(precomputed_features)
    set_shape_fn = update_dataset_shape_wrapper(experts, language_model)

    return [(dataset
        .map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(set_shape_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ) for dataset in datasets]

def get_precomputed_features(source_dataset, experts):
    precomputed_features = []

    for expert in experts:
        precomputed_features.append(
            cache.get_cached_features_by_expert_and_dataset(
                source_dataset, expert))

    return precomputed_features

def generate_encoder_datasets(language_model, source_dataset, experts):
    train_ds = cache.get_cached_language_model_embeddings(
        source_dataset, language_model, "train")

    valid_ds = cache.get_cached_language_model_embeddings(
        source_dataset, language_model, "valid")

    test_ds = cache.get_cached_language_model_embeddings(
        source_dataset, language_model, "test")

    precomputed_features = get_precomputed_features(source_dataset, experts)

    return match_cached_embeddings_with_experts(language_model, experts,
        precomputed_features, train_ds, valid_ds, test_ds)

def get_unique_wrapper():
    seen_video_ids = set()

    def filter_fn(video_id):
        if video_id in seen_video_ids:
            return False

        seen_video_ids.add(video_id)

        return True

    return lambda video_id, data: (tf.numpy_function(filter_fn, [video_id], tf.bool))

def get_video_and_text_dataset(dataset):
    unique_videos_filter = get_unique_wrapper()

    video_dataset = (dataset
        .map(
            lambda video_id, video_data, text_data: (video_id, video_data),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .filter(unique_videos_filter))
    
    text_dataset = dataset.map(
        lambda video_id, video_data, text_data: (video_id, text_data),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return video_dataset, text_dataset

def build_corresponding(video_dataset, text_dataset):
    video_id_to_index_in_video_dataset = {}

    def build_map(index, video_id):
        video_id_to_index_in_video_dataset[video_id] = index

        return index

    def build_map_wrapper(index, data):
        video_id, video_data = data
        index = tf.numpy_function(build_map, [index, video_id], tf.int64)

        return index, video_data

    def get_index(video_id):
        if video_id not in video_id_to_index_in_video_dataset:
            raise ValueError()

        return video_id_to_index_in_video_dataset[video_id]

    def get_index_wrapper(video_id, text_data):
        index = tf.numpy_function(get_index, [video_id], tf.int64)

        return index, text_data

    video_dataset = video_dataset.enumerate().map(build_map_wrapper)
    text_dataset = text_dataset.map(get_index_wrapper)


    return video_dataset, text_dataset

def make_inference_dataset(dataset):
    video_dataset, text_dataset = get_video_and_text_dataset(dataset)

    return build_corresponding(video_dataset, text_dataset)


def filter_duplicate_video_ids(dataset):
    seen_videos = set()

    def filter_fn(video_id):
        if video_id in seen_videos:
            return False

        seen_videos.add(video_id)

        return True

    return dataset.filter(
        lambda video_id, *data: tf.numpy_function(
            filter_fn, [video_id], tf.bool))

def prepare_dataset_for_encoder(
    dataset, shuffle_buffer, batch_size, one_caption_per_video):
    dataset = dataset.shuffle(shuffle_buffer)

    if one_caption_per_video:
        dataset = filter_duplicate_captions(dataset)

    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
