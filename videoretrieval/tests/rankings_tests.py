"""Tests the ranking metrics code.

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

from metrics import rankings


class RankingMetricsTests(unittest.TestCase):

    def test_ranking_generation(self):
        mock_video_embeddings = [tf.constant([
            [5.0],
            [4.0],
            [3.0],
            [2.0],
            [1.0],
            [0.]])]

        mock_text_embeddings = tf.constant([
            [0.4],
            [4.0],
            [3.1],
            [4.1]
            [0.4],
            [0.0]
            ])

    def test_metric_computations(self):
        # 1 dimensional embeddings is fine for the purpose of testing rankings

        mock_ranks = tf.constant([1, 1, 2, 5, 5, 7, 9, 8, 10, 11, 13, 15])
        k_expected_recall_pairs = [
            (1, 1/6),
            (5, 5/12),
            (10, 9/12),
            (11, 10/12),
            (15, 1.0)]

        self.assertTrue(rankings.get_mean_rank(mock_ranks) == 7.25)
        self.assertTrue(rankings.get_median_rank(mock_ranks) == 8)

        for k, recall in k_expected_recall_pairs:
            self.assertTrue(rankings.get_recall_at_k(mock_ranks, k))

