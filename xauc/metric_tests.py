# Lint as: python3
"""Unit tests for metrics.py

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
import metrics

import numpy as np

class TestCrossAUCMetrics(unittest.TestCase):
    """Unit tests for the xAUC metrics in the metrics file.

    Attrs:
        maximum_error: the maximum allowable difference between the target_score
            and the calculated score.
    """

    maximum_error = 1e-6

    def assert_metric_values(self, metric, target_score, arg_a, arg_b):
        """Asserts that calculated metrics are equal to the target score.

        Note: this method assumes that for arg_a and arg_b, 1 is the desired 
        label 0 is the undesired label.

        Args:
            metric: a function that takes in two inputs, arg_a and arg_b, and
                returns a numerical score.
            target_score: the expected numerical score of metric(arg_a, arg_b). 
            arg_a: a vector that's the first parameter of metric.
            arg_b: a vector that's the second parameter of metric.
        """
        calculated_score = metric(
            arg_a, arg_b, desired_label=1, undesired_label=0)

        self.assertTrue(
            abs(target_score - calculated_score) <= self.maximum_error)

        inverse_arg_a = 1 - arg_a
        inverse_arg_b = 1 - arg_b

        calculated_score = metric(inverse_arg_a, inverse_arg_b,
            desired_label=0, undesired_label=1)

        self.assertTrue(
            abs(target_score - calculated_score) <= self.maximum_error)

    def verify_all_metrics(self, target_score, arg_a, arg_b):
        """Calls assert_metric_values on all metrics defined in metrics.py."""
        # All of the xAUC metrics do the same computation, which is why all of
        # these function calls are the same.

        for metric in [metrics.x_auc, metrics.x_auc_0, metrics.x_auc_1]:
                self.assert_metric_values(metric, target_score, arg_a, arg_b)

    def test_high_xauc_score(self):
        """Tests calculating a high xAUC score with both label arrangements."""
        target_score = 15/16

        scores_a = np.array([0.9, 0.8, 0.7, 0.4])
        scores_b = np.array([0.5, 0.3, 0.2, 0.1])

        self.verify_all_metrics(target_score, scores_a, scores_b)

    def test_low_xauc_score(self):
        """Tests calculating a low xAUC score with both label arrangements."""
        target_score = 7/16

        scores_a = np.array([0.8, 0.7, 0.4, 0.2])
        scores_b = np.array([0.9, 0.6, 0.5, 0.3])

        self.verify_all_metrics(target_score, scores_a, scores_b)

    def test_different_input_sizes(self):
        """Tests calculating an xAUC score with different sizes inputs."""
        target_score = 12/15

        scores_a = np.array([0.8, 0.7, 0.6, 0.5, 0.4])
        scores_b = np.array([0.55, 0.45, 0.35])

        self.verify_all_metrics(target_score, scores_a, scores_b)

        target_score = 11/12

        scores_a = np.array([0.9, 0.8])
        scores_b = np.array([0.85, 0.6, 0.6, 0.6, 0.6, 0.6])

        self.verify_all_metrics(target_score, scores_a, scores_b)


if __name__ == "__main__":
    unittest.main()
