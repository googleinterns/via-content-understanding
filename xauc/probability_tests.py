# Lint as: python3
"""Tests the probability functions in probability.py.

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
import numpy as np
import probability

class ProbabilityFunctionsTest(unittest.TestCase):
    """Unit tests for functions in probability.py.

    Attrs:
        maximum_error: upper bound on difference between expected and calculated
            value.
    """

    maximum_error = 1e-6

    def test_probability_functions_unordered_different_sizes(self):
        """Tests with unordered/differently sized input vectors."""
        target_prob = 15/20

        scores_a = np.array([0.7, 0.15, 0.5, 0.25, 0.3])
        scores_b = np.array([0.2, 0.4, 0.2, 0.1])

        calculated = \
            probability.prob_scores_desired_ranked_above_scores_undesired(
                scores_a, scores_b, 1, 0)

        self.assertTrue(abs(calculated - target_prob) < self.maximum_error)

        # The probability should be flipped when the labels are flipped
        target_prob = 1 - target_prob

        calculated = \
            probability.prob_scores_desired_ranked_above_scores_undesired(
                scores_a, scores_b, 0, 1)

        self.assertTrue(abs(calculated - target_prob) < self.maximum_error)

    def test_probability_functions_ordered_same_size(self):
        """Tests with ordered/same sized input vectors."""
        target_prob = 11/16

        scores_a = np.array([0.7, 0.65, 0.5, 0.05])
        scores_b = np.array([0.6, 0.3, 0.2, 0.1])

        calculated = \
            probability.prob_scores_desired_ranked_above_scores_undesired(
                scores_a, scores_b, 1, 0)

        self.assertTrue(abs(calculated - target_prob) < self.maximum_error)

        # The probability should be flipped when the labels are flipped
        target_prob = 1 - target_prob

        calculated = \
            probability.prob_scores_desired_ranked_above_scores_undesired(
                scores_a, scores_b, 0, 1)

        self.assertTrue(abs(calculated - target_prob) < self.maximum_error)


if __name__ == "__main__":
    unittest.main()
