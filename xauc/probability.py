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
from sklearn import metrics

from labels import validate_labels

class ProbabilityCalculator:

    def __init__(self, preferred_outcome_label, undesired_outcome_label):
        """Initializes an instance of ProbabilityCalculator with a   

        Arguments:
            preferred_outcome_label:
            undesired_outcome_label: 
        """
        validate_labels(preferred_outcome_label, undesired_outcome_label)

        self.preferred_outcome_label = preferred_outcome_label
        self.undesired_outcome_label = undesired_outcome_label

    def probability_preferred_ranked_above_undesired(
        self, preferred_scores, undesired_scores):
        """Return the probability that preferred scores are above undesired ones.

        This method uses sklearn's AUC function to calculate the probability
        that a score in preferred_scores is more preferred with 
        respect to the labels self.preferred_outcome_label and the label 
        self.undesired_outcome_label than a score in undesired_score. 

        The scores vector inputted to the AUC function is the concatenation of
        preferred_scores and undesired_scores, while the labels vector inputted
        is self.preferred_outcome_label for the first len(preferred_scores)
        items and self.undesired_outcome_label for the rest of the
        vector. The value returned by the AUC function is the return value.


        Arguments:
            self: an instance of ProbabilityCalculator.
            preferred_scores: scores of items with a ground truth label of a
                preferred outcome.
            undesired_scores: scores of items with a ground truth label of an
                undesired outcome. 
        """

        scores = np.concatenate((preferred_scores, undesired_scores))
        labels = np.zeros_like(scores)

        preferred_scores_length = len(preferred_scores)

        labels[:preferred_scores_length] = self.preferred_outcome_label
        labels[preferred_scores_length:] = self.undesired_outcome_label

        probability = metrics.roc_auc_score(labels, scores)

        return probability
