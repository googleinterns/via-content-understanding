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
    """Unit tests for the xAUC metrics in the metrics module.

    Attrs:
        maximum_error: the maximum allowable difference between the target_score
            and the calculated score.
    """

    maximum_error = 1e-6

    def verify_metrics_match(self, expected, calculated):
        """Verify the calculated and expected metrics match."""

        self.assertTrue(
            abs(expected.xauc - calculated.xauc) < self.maximum_error)

        self.assertTrue(
            abs(expected.xauc0 - calculated.xauc0) < self.maximum_error)

        self.assertTrue(
            abs(expected.xauc1 - calculated.xauc1) < self.maximum_error)

    def verify_metrics(
        self, protected_positive_scores, protected_negative_scores,
        expected_protected_metrics, other_positive_scores,
        other_negative_scores, expected_other_metrics):
        """Asserts that calculated metrics are equal to the expected metrics.

        Note: this method assumes that for arg_a and arg_b, 1 is the desired 
        label 0 is the undesired label.

        Args:
            metric: a function that takes in two inputs, arg_a and arg_b, and
                returns a numerical score.
            target_score: the expected numerical score of metric(arg_a, arg_b). 
            arg_a: a vector that's the first parameter of metric.
            arg_b: a vector that's the second parameter of metric.
        """
        protected_metrics, other_metrics = metrics.calculate_xauc_metrics(
            protected_positive_scores, protected_negative_scores,
            other_positive_scores, other_negative_scores, 1, 0)

        self.verify_metrics_match(protected_metrics, expected_protected_metrics)
        self.verify_metrics_match(other_metrics, expected_other_metrics)
    
    def test_biased_scores(self):
        """Tests calculating xAUC scores for biased scores."""

        protected_positive_scores = np.array([0.75, 0.64, 0.46])
        protected_negative_scores = np.array([0.33, 0.21, 0.10])

        other_positive_scores = np.array([0.83, 0.81, 0.61, 0.58])
        other_negative_scores = np.array([0.74, 0.41, 0.49, 0.44])

        expected_protected_metrics = metrics.xAUCMetrics(
            xauc=9/12, xauc0=1, xauc1=18/21)

        expected_other_metrics = metrics.xAUCMetrics(
            xauc=1.0, xauc0=23/28, xauc1=26/28)

        self.verify_metrics(protected_positive_scores,
            protected_negative_scores, expected_protected_metrics,
            other_positive_scores, other_negative_scores,
            expected_other_metrics)

    def test_fair_scores(self):
        """Tests calculating xAUC scores for fair scores."""

        protected_positive_scores = np.array([0.96, 0.8, .4])
        protected_negative_scores = np.array([0.6, 0.33, 0.21])

        other_positive_scores = np.array([0.83, 0.94, .30])
        other_negative_scores = np.array([0.7, 0.3, 0.2])

        expected_protected_metrics = metrics.xAUCMetrics(
            xauc=8/9, xauc0=15/18, xauc1=16/18)

        expected_other_metrics = metrics.xAUCMetrics(
            xauc=7/9, xauc0=15/18, xauc1=14/18)

        self.verify_metrics(protected_positive_scores,
            protected_negative_scores, expected_protected_metrics,
            other_positive_scores, other_negative_scores,
            expected_other_metrics)


if __name__ == "__main__":
    unittest.main()
