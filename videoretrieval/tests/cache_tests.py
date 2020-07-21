"""Unit tests for the caching module.

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
import tensorflow as tf
from cache import language_model_cache

class TestLanguageModelCache(unittest.TestCase):
    """Tests non i/o functions in the language models cache package."""

    MOCK_NUM_TOKENS = 5
    MOCK_EMBEDDING_DIM = 10
    MOCK_VIDEO_ID = tf.constant("mock_video_id")

    deserialize = staticmethod(language_model_cache.unserialize_data_wrapper(
        MOCK_NUM_TOKENS, MOCK_EMBEDDING_DIM))

    def serialize_data(self, data, num_tokens):
        return language_model_cache.serialize_to_protobuf(
            self.MOCK_VIDEO_ID, data, num_tokens)

    def assertTensorsEqual(self, tensor_a, tensor_b):
        self.assertTrue(tf.reduce_all(tensor_a == tensor_b) == True)

    def test_serialization_loop_normal_data(self):
        """Tests serializing and deserializing without truncation or padding."""
        tf.random.set_seed(1)

        mock_embeddings = tf.random.normal(
            (1, self.MOCK_NUM_TOKENS, self.MOCK_EMBEDDING_DIM))

        serialized = self.serialize_data(
            mock_embeddings, self.MOCK_NUM_TOKENS)
        deserialized_video_id, deserialized_data = self.deserialize(serialized)

        self.assertEqual(deserialized_video_id, self.MOCK_VIDEO_ID)
        self.assertTensorsEqual(deserialized_data, mock_embeddings[0])

    def test_serialization_loop_truncating_data(self):
        """Tests serializing and deserializing with truncation."""
        tf.random.set_seed(1)

        mock_embeddings = tf.random.normal(
            (1, self.MOCK_NUM_TOKENS + 1, self.MOCK_EMBEDDING_DIM))

        serialized = self.serialize_data(
            mock_embeddings, self.MOCK_EMBEDDING_DIM)
        deserialized_video_id, deserialized_data = self.deserialize(serialized)

        self.assertEqual(deserialized_video_id, self.MOCK_VIDEO_ID)
        self.assertTensorsEqual(
            mock_embeddings[0, :self.MOCK_NUM_TOKENS], deserialized_data)

    def test_serialization_loop_padding_data(self):
        """Tests serializing and deserializing with padding."""
        tf.random.set_seed(1)

        mock_embeddings = tf.random.normal(
            (1, self.MOCK_NUM_TOKENS - 1, self.MOCK_EMBEDDING_DIM))

        serialized = self.serialize_data(
            mock_embeddings, self.MOCK_EMBEDDING_DIM)
        deserialized_video_id, deserialized_data = self.deserialize(serialized)

        self.assertEqual(deserialized_video_id, self.MOCK_VIDEO_ID)
        self.assertTensorsEqual(
            mock_embeddings, deserialized_data[:self.MOCK_NUM_TOKENS - 1])
        self.assertTensorsEqual(
            deserialized_data[self.MOCK_NUM_TOKENS + 1:],
            tf.zeros_like(deserialized_data[self.MOCK_NUM_TOKENS + 1:]))
