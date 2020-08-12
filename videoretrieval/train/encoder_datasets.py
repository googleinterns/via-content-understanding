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
import enum

class TrainingDatasetType(enum.Enum):
    """An enum that indicates what type of text data the dataset holds."""
    EmbeddingsDataset = 0
    EncodingsDataset = 1

class MapFunctionWrapper:
    """A wrapper for functions that operate on different types of datasets.""" 
    def __init__(self, embeddings_function, encodings_function):
        """Initializes the wrapper.

        Args:
            embeddings_function: a map function that operates on embeddings
                datasets.
            encodings_function: a map function that operates on encoding
                datasets.
        """
        self.embeddings_function = embeddings_function
        self.encodings_function = encodings_function

def get_map_function(map_function_wrapper, training_dataset_type):
    """Gets the correct map function for a given dataset type.

    Args:
        map_function_wrapper: an object of type MapFunctionWrapper.
        training_dataset_type: a value from the TrainingDatasetType enum that
            indicates the type of dataset that map function should be for.

    Returns: a function that operates on the correct dataset type.

    Raises:
        Value Error: a `ValueError` is raised if the training_dataset_type is
            invalid. 
    """  
    if training_dataset_type == TrainingDatasetType.EmbeddingsDataset:
        return map_function_wrapper.embeddings_function
    elif training_dataset_type == TrainingDatasetType.EncodingsDataset:
        return map_function_wrapper.encodings_function
    else:
        raise ValueError()


def replace_video_id_with_expert_features_wrapper(precomputed_features):
    """Returns a function that adds precomputed features to an example.

    Arguments:
        precomputed_features: an array of dicts, where each dict maps from a
            video id to a tuple. The tuple has two elements, the first being the
            data of the precomputed feature, the second being a boolean that
            indicates if the feature is missing.

    Returns: a MapFunctionWrapper for a map function that takes
        a video_id, which is a video id as a string, and ids, the contextual
        embeddings for the given caption. This function then returns a tuple of
        video_id, expert_features, the contextual embeddings, an attention mask
        if the dataset has attention masks, and a tensor of missing expert
        modalities.
    """
    output_shape = (tf.bool,) + len(precomputed_features) * (tf.float32,)

    def get_expert_features(video_id_encoded):
        """Gets the features for a given video id"""
        expert_features = []
        missing_modalities = []

        video_id = video_id_encoded.decode("utf-8")

        for feature_dict in precomputed_features:
            assert video_id in feature_dict, f"Video {video_id} missing data"
            features, data_exists = feature_dict[video_id]
            expert_features.append(features)
            missing_modalities.append(data_exists)

        return [np.array(missing_modalities)] + expert_features

    def embeddings_wrapper(video_id, ids):  
        expert_data = tf.numpy_function(
            get_expert_features, [video_id], output_shape)

        missing_modalities = expert_data[0]
        expert_features = expert_data[1:]

        return (video_id, tuple(expert_features), ids, missing_modalities)

    def encodings_wrapper(video_id, encoded, attention_mask):
        expert_data = tf.numpy_function(
            get_expert_features, [video_id], output_shape)

        missing_modalities = expert_data[0]
        expert_features = expert_data[1:]

        return (
            video_id,
            tuple(expert_features),
            encoded,
            attention_mask,
            missing_modalities)

    return MapFunctionWrapper(
        embeddings_function=embeddings_wrapper,
        encodings_function=encodings_wrapper)

def update_dataset_shape_wrapper(experts, language_model):
    """Updates the shapes of expert features and text embeddings for a dataset.

    Arguments:
        experts: a list of experts (type BaseExpert) used in the dataset.
        language_model: a language model of type BaseLanguageModel that was used
            to generate contextual embeddings. 

    Returns: a function wrapped by MapFunctionWrapper that takes data for a
        given dataset as parameters and updates the shape of the tensors. 
    """
    num_experts = len(experts)
    expert_shapes = [expert.embedding_shape for expert in experts]
    contextual_embeddings_shape = language_model.contextual_embeddings_shape

    def embeddings_map_fn(
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

    def encodings_map_fn(
        video_id, expert_features, encodings, attention_mask,
        missing_modalities):
        for expert_feature, shape in zip(expert_features, expert_shapes):
            expert_feature.set_shape(shape)

        encodings.set_shape(
            (language_model.contextual_embeddings_shape[0],))
        missing_modalities.set_shape(num_experts)
        attention_mask.set_shape(
            (language_model.contextual_embeddings_shape[0],))

        return (
            video_id,
            expert_features,
            encodings,
            attention_mask,
            missing_modalities)

    return MapFunctionWrapper(
        embeddings_function=embeddings_map_fn,
        encodings_function=encodings_map_fn)

def match_cached_embeddings_with_experts(
    language_model, experts, precomputed_features, datasets,
    training_dataset_type):
    """Matches items in a dataset with the precomputed features

    Parameters:
        language_model: the language model the contextual embeddings are from.
        experts: a list of experts that the precomputed features are from.
        precomputed_features: a list of dicts, one per expert, that map from
          video id to precomputed feature.
        datasets: the datasets to transform.

    Returns: A list of tf.data Datasets, where each example in the dataset
        consists of: a video id, precomputed features,
        contextual embeddings or encodings, potentially an attention mask
        (depending on the dataset type), and a boolean
        vector that indicates which expert modalities are missing.  
    """
    match_features_fn = get_map_function(
        replace_video_id_with_expert_features_wrapper(precomputed_features),
        training_dataset_type)
    set_shape_fn = get_map_function(
        update_dataset_shape_wrapper(experts, language_model),
        training_dataset_type) 

    return [(dataset
        .map(
            match_features_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(
            set_shape_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ) for dataset in datasets]

def is_expert_value_missing(expert_value):
    """Returns if a given expert_value is missing or not."""
    return type(expert_value) == float and np.isnan(expert_value)

def zero_pad_expert_features(expert, expert_value):
    """Zero pads a variable length precomputed feature."""
    frames = expert_value.shape[0]

    if frames >= expert.max_frames:
        video_expert_features = \
            expert_value[:expert.max_frames]
    else:
        zero_padding = np.zeros((
            expert.max_frames - frames,
            *expert.embedding_shape[1:]), np.float32)

        video_expert_features = np.concatenate((
            expert_value, zero_padding))

    return video_expert_features

def get_precomputed_features(source_dataset, experts):
    """Get precomputed features from a set of experts and a dataset.

    Arguments:
        source_dataset: the source dataset as an instance of Base Dataset.
        experts: a list of experts to use precomputed features from

    Returns: A list of dicts, where each dict maps from video id to precomputed
        features. 
    """
    
    precomputed_features = []

    for expert in experts:
        processed_expert_features = {}

        expert_features = cache.get_cached_features_by_expert_and_dataset(
            source_dataset, expert)

        for video_id, expert_value in expert_features.items():
            video_expert_features = None
            missing_modalities = False

            expert_value = expert.feature_transformation(expert_value)

            if is_expert_value_missing(expert_value):
                video_expert_features = np.zeros(
                    expert.embedding_shape, np.float32)
                missing_modalities = True
            else:
                expert_value = expert_value.astype(np.float32)
                
                if expert.constant_length:
                    video_expert_features = expert_value
                else:
                    video_expert_features = zero_pad_expert_features(
                        expert, expert_value)

            processed_expert_features[video_id] = (
                video_expert_features,
                missing_modalities
            )

        precomputed_features.append(processed_expert_features)

    return precomputed_features

def sample_captions(ds, captions_per_video, dataset_type):
    """Given a dataset, samples one caption per video."""
    def random_index(sample):
        return np.random.randint(0, sample.shape[0]) 

    def sample_captions_embeddings(video_ids_batch, contextual_embeddings_batch):
        index = tf.numpy_function(
            random_index,
            [contextual_embeddings_batch],
            tf.int64)

        return video_ids_batch[index], contextual_embeddings_batch[index]

    def sample_captions_encodings(video_ids_batch, encodings, attention_masks):
        index = tf.numpy_function(random_index, [encodings], tf.int64)
        return video_ids_batch[index], encodings[index], attention_masks[index]
    
    map_fn = get_map_function(
        MapFunctionWrapper(
            sample_captions_embeddings, sample_captions_encodings),
        dataset_type)

    return ds.batch(captions_per_video).map(map_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

def generate_encoder_datasets(language_model, source_dataset, experts):
    """Generates datasets necessary to train encoders.

    Args:
        language_model: an instance of BaseLanguageModel who's embeddings should
            be use.
        source_dataset: The dataset to generate the embeddings from. 
        experts: The experts to be used.

    Returns: A tuple of three tf.data datasets: the train dataset, the
        validation dataset, and the test dataset. Each element of each of these
        datasets is a tuple where the first element is the video ids, the second
        element is the precomputed video features, the third is the contextual
        embeddings, and the fourth a boolean tensor of missing video expert
        modalities.  
    """
    train_ds = cache.get_cached_language_model_embeddings(
        source_dataset, language_model, "train")

    valid_ds = cache.get_cached_language_model_embeddings(
        source_dataset, language_model, "valid")

    test_ds = cache.get_cached_language_model_embeddings(
        source_dataset, language_model, "test")

    return generate_dataset(
        language_model=language_model,
        experts=experts,
        source_dataset=source_dataset,
        dataset_type=TrainingDatasetType.EmbeddingsDataset,
        dataset_splits=[train_ds, valid_ds, test_ds],
        splits_to_sample=[0])

def generate_language_model_fine_tuning_datasets(
    language_model, source_dataset, experts):
    """Generates datasets for training encoders and fine tuning a language model.

    Args:
        language_model: an instance of BaseLanguageModel who's embeddings should
            be use.
        source_dataset: The dataset to generate the embeddings from. 
        experts: The experts to be used.

    Returns: A tuple of three tf.data datasets: the train dataset, the
        validation dataset, and the test dataset. Each element of each of these
        datasets is a tuple where the first element is the video ids, the second
        element is the precomputed video features, the third is the encoded
        text, the fourth is the attention masks for the encoded text, and the 
        fifth is the boolean tensor of missing video expert modalities.
    """
    train_ds = cache.get_cached_language_model_encodings(
        source_dataset, language_model, "train").cache()

    valid_ds = cache.get_cached_language_model_encodings(
        source_dataset, language_model, "valid").cache()

    test_ds = cache.get_cached_language_model_encodings(
        source_dataset, language_model, "test").cache()

    return generate_dataset(
        language_model=language_model,
        experts=experts,
        source_dataset=source_dataset,
        dataset_type=TrainingDatasetType.EncodingsDataset,
        dataset_splits=[train_ds, valid_ds, test_ds],
        splits_to_sample=[0])

def generate_dataset(
    language_model, experts, source_dataset, dataset_type, dataset_splits,
    splits_to_sample):
    """Generates a dataset for training.

    Args:
        language_model: the language model that is going to be used for this
            model.
        experts: the experts to be used.
        source_dataset: the dataset to generate embeddings from.
        dataset_type: the dataset type (an option from TrainingDatasetType).
        dataset_splits: the cached text data for each split in the dataset.
        splits_to_sample: indexes of items in dataset_splits that captions will
            be sampled from. 
    """
    for split_index in splits_to_sample:
        dataset_splits[split_index] = sample_captions(
            ds=dataset_splits[split_index],
            captions_per_video=source_dataset.captions_per_video,
            dataset_type=dataset_type)

    precomputed_features = get_precomputed_features(source_dataset, experts)

    return match_cached_embeddings_with_experts(
        language_model, experts, precomputed_features, dataset_splits,
        dataset_type)
