"""Unit tests for the encoder datasets module.

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

import unittest
from train import encoder_datasets
import tensorflow as tf
import numpy as np
from base import BaseExpert, BaseLanguageModel

MOCK_TENSOR_SHAPE = (2, 2) 
MOCK_TENSOR_DATA = tf.zeros(MOCK_TENSOR_SHAPE)
NUM_MOCK_VIDEOS = 5
MOCK_CAPTIONS_PER_VIDEO = 20

class MockExpert(BaseExpert):
    """An implementation of BaseExpert used for unit tests."""
    def __init__(self, is_constant_length, max_frames, embedding_shape):
        self.constant_length = is_constant_length
        self.max_frames = max_frames
        self._embedding_shape = embedding_shape

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def embedding_shape():
        return self._embedding_shape

class MockLanguageModel(BaseLanguageModel):
    """An implementation of BaseLanguageModel used for unit tests."""

    def __init__(self, contextual_embeddings_shape):
        self.contextual_embeddings_shape = contextual_embeddings_shape

    def forward(self, ids):
        raise NotImplementedError()

    def encode(self, text):
        raise NotImplementedError()

    @property
    def batch_size(self):
        raise NotImplementedError()

def make_mock_video_ids(num_videos):
    """Makes a list of video ids used for unit tests.

    num_videos: an integer; the number of mock video ids to make. 

    Returns: a list of strings of that can be used as video ids for unit tests.
    """
    video_ids = []

    for video_num in range(num_videos):
        video_ids.append(f"video{video_num}")

    return video_ids

def mock_id_embeddings_pair_dataset_generator(
    video_ids, mock_tensor_shape, mock_embeddings_per_video):
    for video_id in video_ids:
        for _ in range(mock_embeddings_per_video):
            yield (video_id, tf.zeros(mock_tensor_shape))

def make_mock_id_embeddings_pair_dataset(
    video_ids, mock_tensor_shape, mock_embeddings_per_video=1):
    
    dataset_generator = lambda: mock_id_embeddings_pair_dataset_generator(
        video_ids, mock_tensor_shape, mock_embeddings_per_video)

    return tf.data.Dataset.from_generator(dataset_generator, 
        (tf.string, tf.float32))

def make_mock_precomputed_features(video_ids):
    mock_precomputed_features_available = {}
    mock_precomputed_features_half_missing = {}

    for index, video_id in enumerate(video_ids):
        mock_precomputed_features_available[video_id] = (
            index + MOCK_TENSOR_DATA, True)
        mock_precomputed_features_half_missing[video_id] = (
            -1 * index + MOCK_TENSOR_DATA, index % 2 == 0)

    return [
        mock_precomputed_features_available,
        mock_precomputed_features_half_missing]

class TestEncoderDatasetsFunctions(unittest.TestCase):
    """"Tests for functions in the encoder datasets module."""
    mock_video_ids = make_mock_video_ids(NUM_MOCK_VIDEOS)
    mock_dataset = make_mock_id_embeddings_pair_dataset(
        mock_video_ids, MOCK_TENSOR_SHAPE)
    mock_precomputed_features = make_mock_precomputed_features(
        mock_video_ids)

    def test_replacing_video_id_with_expert_features(self):
        """Tests the replace_video_id_with_expert_features_wrapper function."""
        map_fn = encoder_datasets.replace_video_id_with_expert_features_wrapper(
            self.mock_precomputed_features)

        output = list(iter(mock_dataset.map(map_fn)))

        for video_id, expert_features, _, missing_modalities in output:
            video_id = video_id.numpy().decode("utf-8")

            for feature_index, (feature, missing) in enumerate(zip(
                expert_features, missing_modalities)):

                expected_feature, expected_missing = \
                    self.mock_precomputed_features[feature_index][video_id]
                
                self.assertTrue(np.array_equal(
                    feature.numpy(), expected_feature.numpy()))
                self.assertEqual(missing, expected_missing)


    def test_update_dataset_shape_wrapper(self):
        pass

    def test_zero_pad_expert_features(self):
        mock_expert = MockExpert(False, 10, 10)

    def test_sample_captions(self):
        """Tests the sample captions method of encoder datasets."""
        mock_dataset = list(iter(encoder_datasets.sample_captions(
            self.mock_dataset, MOCK_CAPTIONS_PER_VIDEO)))

        for expected_value, dataset_value in zip(
            self.mock_video_ids, mock_dataset):
            video_id, _ = dataset_value

            self.assertEqual(expected_value, video_id)


