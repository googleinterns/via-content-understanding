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
    output_shape = (tf.bool,) + len(precomputed_features) * (tf.float32,)

    def get_expert_features(video_id_encoded):
        expert_features = []
        missing_modalities = []

        video_id = video_id_encoded.decode("utf-8")

        for feature_dict in precomputed_features:
            features, data_exists = feature_dict[video_id]
            expert_features.append(features)

            missing_modalities.append(data_exists)

        return [np.array(missing_modalities)] +  expert_features

    def wrapper(video_id, ids):
        expert_data = tf.numpy_function(
            get_expert_features, [video_id], output_shape)

        missing_modalities = expert_data[0]
        expert_features = expert_data[1:]

        return (video_id, tuple(expert_features), ids, missing_modalities)


    return wrapper

def update_dataset_shape_wrapper(experts, language_model):
    num_experts = len(experts)
    expert_shapes = [expert.embedding_shape for expert in experts]
    contextual_embeddings_shape = language_model.contextual_embeddings_shape

    def map_fn(
        video_id, expert_features, contextual_embeddings, missing_modalities):
        for expert_feature, shape in zip(expert_features, expert_shapes):
            expert_feature.set_shape(shape)

        contextual_embeddings.set_shape(contextual_embeddings_shape)
        missing_modalities.set_shape(num_experts)

        return (
            video_id, 
            expert_features,
            contextual_embeddings,
            missing_modalities)

    return map_fn


def match_cached_embeddings_with_experts(
    language_model, experts, precomputed_features, *datasets):
    """Matches the cached"""
    map_fn = replace_video_id_with_expert_features_wrapper(precomputed_features)
    set_shape_fn = update_dataset_shape_wrapper(experts, language_model)

    return [(dataset
        .map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(set_shape_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ) for dataset in datasets]

def get_precomputed_features(source_dataset, experts):
    """Get precomputed features from a set of experts and a dataset.

    Arguments:
        source_dataset: 
        experts:

    Returns:
    """

    
    precomputed_features = []

    for expert in experts:
        processed_expert_features = {}

        expert_features = cache.get_cached_features_by_expert_and_dataset(
            source_dataset, expert)

        for video_id, expert_value in expert_features.items():
            video_expert_features = None
            missing_modalities = False

            if expert.name == "densenet":
                expert_value = expert_value[0]

            if type(expert_value) == float and np.isnan(expert_value):
                video_expert_features = np.zeros(
                    expert.embedding_shape, np.float32)
                missing_modalities = True
            else:
                expert_value = expert_value.astype(np.float32)
                if expert.constant_length:
                    video_expert_features = expert_value
                else:
                    frames = expert_value.shape[0]

                    if frames >= expert.max_frames:
                        video_expert_features = \
                            expert_value[:expert.max_frames]
                    else:
                        video_expert_features = np.concatenate((
                            expert_value, np.zeros((
                                expert.max_frames - frames,
                                *expert.embedding_shape[1:]), np.float32)))

            processed_expert_features[video_id] = (
                video_expert_features,
                missing_modalities
            )

        precomputed_features.append(processed_expert_features)

    return precomputed_features

def sample_captions(ds):
    def random_index(sample):
        return np.random.randint(0, sample.shape[0]) 

    def sample_caption_wrapper(video_ids_batch, contextual_embeddings_batch):
        index = tf.numpy_function(
            random_index,
            [contextual_embeddings_batch],
            tf.int64)

        return video_ids_batch[index], contextual_embeddings_batch[index, :, :]
    
    return ds.batch(20).map(sample_caption_wrapper,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

def generate_encoder_datasets(language_model, source_dataset, experts):
    """Generates datasets necessary to train encoders.

    Arguments:
        language_model: an instance of BaseLanguageModel
        source_dataset:
        experts:

    Returns:

    """
    train_ds = cache.get_cached_language_model_embeddings(
        source_dataset, language_model, "train")

    valid_ds = cache.get_cached_language_model_embeddings(
        source_dataset, language_model, "valid")

    test_ds = cache.get_cached_language_model_embeddings(
        source_dataset, language_model, "test")

    train_ds = sample_captions(train_ds)

    precomputed_features = get_precomputed_features(source_dataset, experts)

    return match_cached_embeddings_with_experts(language_model, experts,
        precomputed_features, train_ds, valid_ds, test_ds)