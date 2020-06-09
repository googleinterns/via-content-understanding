# Lint as: python3
"""Functions for computing probabilities used for calculating xAUC metrics.

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

import numpy as np
from labels import validate_labels

class ProbabilityCalculator:
    """A class for calculating probabilities."""

    def __init__(self, preferred_outcome_label, undesired_outcome_label):
        """Initializes an instance of ProbabilityCalculator

        Arguments:
            preferred_outcome_label: the label for preferred outcomes.
            undesired_outcome_label: the label for undesired outcomes.
        """
        validate_labels(preferred_outcome_label, undesired_outcome_label)

        self.preferred_outcome_label = preferred_outcome_label
        self.undesired_outcome_label = undesired_outcome_label

        self.preferred_are_positive = \
            self.preferred_outcome_label > self.undesired_outcome_label

        if self.preferred_are_positive:
            self.preferred_are_ranked_above_undesired = lambda a, b: a > b
        else:
            self.preferred_are_ranked_above_undesired = lambda a, b: a < b

    def probability_preferred_ranked_above_undesired(
        self, preferred_scores, undesired_scores):
        """Finds the probability that preferred scores are above undesired ones.


        Arguments:
            self: an instance of ProbabilityCalculator.
            preferred_scores: scores of items with a ground truth label of a
                preferred outcome.
            undesired_scores: scores of items with a ground truth label of an
                undesired outcome. 
        """

        preferred_examples = len(preferred_scores)
        undesired_examples = len(undesired_scores)

        if preferred_examples == 0 or undesired_examples == 0:
            raise ValueError(
                "Preferred scorse or undesired scores had no length")

        preferred_scores = np.sort(preferred_scores)
        undesired_scores = np.sort(undesired_scores)

        if self.preferred_are_positive:
            preferred_scores = preferred_scores[::-1]
            undesired_scores = undesired_scores[::-1]
        
        undesired_index = 0

        preferred_scores_ranked_above_undesired = 0

        for preferred_index in range(preferred_examples):

            while undesired_index < undesired_examples and \
                not self.preferred_are_ranked_above_undesired(
                    preferred_scores[preferred_index],
                    undesired_scores[undesired_index]):

                undesired_index += 1


            preferred_scores_ranked_above_undesired += \
                undesired_examples - undesired_index

        return preferred_scores_ranked_above_undesired /\
            (preferred_examples * undesired_examples)
