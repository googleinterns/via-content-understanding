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
            mock_perfect_video_embeddings,
            mock_perfect_text_embeddings,
            mock_mixture_weights,
            mock_missing_experts,
            1.0)

        self.assertTrue(abs(loss.numpy() - 0.0) < self.error)

        loss = bidirectional_max_margin_ranking_loss(
            mock_perfect_video_embeddings,
            mock_perfect_text_embeddings,
            mock_mixture_weights,
            mock_missing_experts,
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
            mock_good_video_embeddings,
            mock_good_text_embeddings,
            mock_mixture_weights,
            mock_missing_experts,
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
            [0, 1.0],
            [-1.0, 1.0],
            [0.7, 0.6],
        ])]

        loss = bidirectional_max_margin_ranking_loss(
            mock_bad_video_embeddings,
            mock_bad_text_embeddings,
            mock_mixture_weights,
            mock_missing_experts,
            1.5)

        self.assertTrue(abs(loss.numpy() - 1.21000000333) < self.error)

