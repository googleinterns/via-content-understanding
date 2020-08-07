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

Tests the metrics code.
"""

import unittest
import tensorflow as tf

from metrics.loss import bidirectional_max_margin_ranking_loss
from metrics.loss import build_similarity_matrix

class TestBidirectionalMaxMarginRankingLoss(unittest.TestCase):
    """Tests bidirectional max margin ranking loss."""

    error = 1e-5


    def test_loss_calculation_with_different_embeddings(self):
        """Tests computing loss for mini-batches of varying quality embeddings.

        This tests computing loss for three sets of embeddings. The first set of
        embeddings are perfect, the second are ok, and the third are bad.
        """

        mock_missing_experts = tf.constant([[False], [False], [False]])
        mock_mixture_weights = tf.constant([[1.0], [1.0], [1.0]])

        mock_perfect_video_embeddings = [tf.constant([
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])]

        mock_perfect_text_embeddings = [tf.constant([
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])]

        loss = bidirectional_max_margin_ranking_loss(
            build_similarity_matrix(
                mock_perfect_video_embeddings,
                mock_perfect_text_embeddings,
                mock_mixture_weights,
                mock_missing_experts),
            1.0)

        self.assertTrue(abs(loss.numpy() - 0.0) < self.error)

        loss = bidirectional_max_margin_ranking_loss(
            build_similarity_matrix(
                mock_perfect_video_embeddings,
                mock_perfect_text_embeddings,
                mock_mixture_weights,
                mock_missing_experts),
            100.0)

        self.assertTrue(abs(loss.numpy() - 99.0) < self.error)

        mock_good_video_embeddings = [tf.constant([
            [-0.9938837 ,  0.11043153],
            [-0.70710677,  0.70710677],
            [0.0, 1.0],
        ])]

        mock_good_text_embeddings = [tf.constant([
            [-1.0, 0.0],
            [-0.5547002,  0.8320503],
            [0.0, 1.0],
        ])]

        loss = bidirectional_max_margin_ranking_loss(
            build_similarity_matrix(
                mock_good_video_embeddings,
                mock_good_text_embeddings,
                mock_mixture_weights,
                mock_missing_experts),
            1.0)

        self.assertTrue(abs(loss.numpy() - 0.5084931) < self.error)


        mock_missing_experts = tf.constant([[False], [False], [False], [False]])
        mock_mixture_weights = tf.constant([[1.0], [1.0], [1.0], [1.0]])

        mock_bad_video_embeddings = [tf.constant([
            [0.25, 0.25],
            [1.0, 1.0],
            [0.6, 0.5],
            [0.9, 0.8],
        ])]

        mock_bad_text_embeddings = [tf.constant([
            [-1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 1.0],
            [0.7, 0.6],
        ])]

        loss = bidirectional_max_margin_ranking_loss(
            build_similarity_matrix(
                mock_bad_video_embeddings,
                mock_bad_text_embeddings,
                mock_mixture_weights,
                mock_missing_experts),
            1.5)

        self.assertTrue(abs(loss.numpy() - 1.21000000333) < self.error)

class TestBuildSimilarityMatrix(unittest.TestCase):
    """Tests the build_similarity_matrix_function in the loss module."""
    mock_batch_size = 5
    mock_embedding_dimensionality = 10
    error = 1e-6

    def assert_matricies_are_the_same(self, matrix_a, matrix_b):
        self.assertTrue(tf.reduce_max(tf.abs(matrix_a - matrix_b)) < self.error)

    def test_computing_similarity_one_expert(self):
        """Tests building a similarity matrix when there is only one expert."""
        tf.random.set_seed(1)

        text_embeddings = tf.random.uniform(
            (self.mock_batch_size, self.mock_embedding_dimensionality))
        video_embeddings = tf.random.uniform(
            (self.mock_batch_size, self.mock_embedding_dimensionality))

        mixture_weights = tf.ones((self.mock_batch_size, 1), tf.float32)
        missing_experts = tf.constant([[False]] * self.mock_batch_size)

        computed_matrix = build_similarity_matrix(
            [video_embeddings],
            [text_embeddings],
            mixture_weights,
            missing_experts)

        self.assertTrue(
            computed_matrix.shape == (
                self.mock_batch_size, self.mock_batch_size))

        self.assert_matricies_are_the_same(
            computed_matrix,
            tf.matmul(text_embeddings, video_embeddings, transpose_b=True))
        
    def test_computing_similarity_multiple_experts(self):
        """Tests building a similarity matrix with multiple experts."""
        tf.random.set_seed(2)

        text_embeddings_expert_one = tf.random.uniform(
            (self.mock_batch_size, self.mock_embedding_dimensionality))
        video_embeddings_expert_one = tf.random.uniform(
            (self.mock_batch_size, self.mock_embedding_dimensionality))

        text_embeddings_expert_two = tf.random.uniform(
            (self.mock_batch_size, self.mock_embedding_dimensionality))
        video_embeddings_expert_two = tf.random.uniform(
            (self.mock_batch_size, self.mock_embedding_dimensionality))

        text_embeddings_expert_three = tf.random.uniform(
            (self.mock_batch_size, self.mock_embedding_dimensionality))
        video_embeddings_expert_three = tf.random.uniform(
            (self.mock_batch_size, self.mock_embedding_dimensionality))

        mixture_weights = tf.random.uniform((self.mock_batch_size, 3))
        missing_experts = tf.constant([
            [False, True, False],
            [False, False, False],
            [False, False, True],
            [False, False, False],
            [False, True, True]])

        available_experts_float32 = 1 - tf.cast(missing_experts, tf.float32)


        weights, _ = tf.linalg.normalize(
            mixture_weights[:, None] * available_experts_float32[None, :],
            axis=-1,
            ord=1)

        expert_one_similarity = weights[:, :, 0] * tf.matmul(
            text_embeddings_expert_one,
            video_embeddings_expert_one,
            transpose_b=True)
        expert_two_similarity = weights[:, :, 1] * tf.matmul(
            text_embeddings_expert_two,
            video_embeddings_expert_two,
            transpose_b=True)
        expert_three_similarity = weights[:, :, 2] * tf.matmul(
            text_embeddings_expert_three,
            video_embeddings_expert_three,
            transpose_b=True)

        expected_matrix = (
            expert_one_similarity + expert_two_similarity +
            expert_three_similarity)

        computed_matrix = build_similarity_matrix(
            [
                video_embeddings_expert_one,
                video_embeddings_expert_two,
                video_embeddings_expert_three],
            [
                text_embeddings_expert_one,
                text_embeddings_expert_two,
                text_embeddings_expert_three],
            mixture_weights,
            missing_experts)

        self.assert_matricies_are_the_same(computed_matrix, expected_matrix)

