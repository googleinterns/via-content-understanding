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

    calculator_preferred_positive = probability.ProbabilityCalculator(1, 0)
    calculator_preferred_negative = probability.ProbabilityCalculator(0, 1)

    def assert_probability_calculated_correctly(
        self, preferred_scores, undesired_scores, expected_probability):
        """Assert the calculated probability is equal to the expected value.

        Note: this method assumes that for preferred_scores and
        undesired_scores, 1 is the label for preferred outcomes and 0 is the
        label for undesired outcomes.
            preferred_scores: a vector of preferred_scores.
            undesired_scores: a vector of undesired_scores.
            expected_probability: the expected probability a score in 
                preferred_scores is ranked more preferred than a score in 
                undesired_scores.
        """

        expected_score_positive = (self.calculator_preferred_positive
            .probability_preferred_ranked_above_undesired(
                preferred_scores, undesired_scores))

        self.assertTrue(
            expected_score_positive - expected_probability < self.maximum_error)

        expected_score_negative = (self.calculator_preferred_negative
            .probability_preferred_ranked_above_undesired(
                1 - preferred_scores, 1 - undesired_scores))


    def test_probability_functions_unordered_different_sizes(self):
        """Tests with unordered/differently sized input vectors."""
        target_prob = 15/20

        scores_preferred = np.array([0.7, 0.15, 0.5, 0.25, 0.3])
        scores_undesired = np.array([0.2, 0.4, 0.2, 0.1])

        self.assert_probability_calculated_correctly(
            scores_preferred, scores_undesired, target_prob)

    def test_probability_functions_ordered_same_size(self):
        """Tests with ordered/same sized input vectors."""
        target_prob = 11/16

        scores_preferred = np.array([0.7, 0.65, 0.5, 0.05])
        scores_undesired = np.array([0.6, 0.3, 0.2, 0.1])

        self.assert_probability_calculated_correctly(
            scores_preferred, scores_undesired, target_prob)


    def test_high_probability(self):
        """Tests calculating a probability with both label arrangements."""
        target_prob = 15/16

        scores_preferred = np.array([0.9, 0.8, 0.7, 0.4])
        scores_undesired = np.array([0.5, 0.3, 0.2, 0.1])

        self.assert_probability_calculated_correctly(
            scores_preferred, scores_undesired, target_prob)

    def test_low_probability(self):
        """Tests calculating a probability with both label arrangements."""
        target_prob = 7/16

        scores_preferred = np.array([0.8, 0.7, 0.4, 0.2])
        scores_undesired = np.array([0.9, 0.6, 0.5, 0.3])

        self.assert_probability_calculated_correctly(
            scores_preferred, scores_undesired, target_prob)

    def test_same_values(self):
        """Tests calculating an probabilites with identical scores."""
        target_prob = 10/15

        scores_preferred = np.array([0.8, 0.7, 0.6, 0.5, 0.4])
        scores_undesired = np.array([0.6, 0.5, 0.35])

        self.assert_probability_calculated_correctly(
            scores_preferred, scores_undesired, target_prob)

    def test_invalid_probabilites(self):
        with self.assertRaises(ValueError):
            (self.calculator_preferred_positive
                .probability_preferred_ranked_above_undesired(
                    np.array([]), np.array([])))

if __name__ == "__main__":
    unittest.main()
