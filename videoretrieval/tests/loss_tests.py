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

    def test_perfect_embeddings(self):
        """Tests a mini batch of perfect embeddings."""
        mock_video_embeddings = tf.Variable([
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])

        mock_text_embeddings = tf.Variable([
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])

        loss = bidirectional_max_margin_ranking_loss(mock_video_embeddings,
            mock_text_embeddings, 1.0)

        self.assertTrue(abs(loss.numpy() - 0.0) < self.error)

        loss = bidirectional_max_margin_ranking_loss(mock_video_embeddings,
            mock_text_embeddings, 100.0)

        self.assertTrue(abs(loss.numpy() - 132.0) < self.error)

    def test_good_embeddings(self):
        """Tests a minibatch of good embeddings."""
        mock_video_embeddings = tf.Variable([
            [-0.9938837 ,  0.11043153],
            [-0.70710677,  0.70710677],
            [0.0, 1.0],
        ])

        mock_text_embeddings = tf.Variable([
            [-1.0, 0.0],
            [-0.5547002,  0.8320503],
            [0.0, 1.0],
        ])

        loss = bidirectional_max_margin_ranking_loss(mock_video_embeddings,
            mock_text_embeddings, 1.0)

        expected_value = 0.6779908

        self.assertTrue(abs(loss.numpy() - 0.6779908) < self.error)

    def test_bad_embeddings(self):
        """Tests a mini batch of bad embeddings."""

        mock_video_embeddings = tf.Variable([
            [0.25, 0.25],
            [1.0, 1.0],
            [0.6, 0.5],
            [0.9, 0.8],
        ])

        mock_text_embeddings = tf.Variable([
            [-1.0, 0.0],
            [0, 1.0],
            [-1.0, 1.0],
            [0.7, 0.6],
        ])

        loss = bidirectional_max_margin_ranking_loss(mock_video_embeddings,
            mock_text_embeddings, 1.5)

        self.assertTrue(abs(loss.numpy() - 1.815000005) < self.error)
