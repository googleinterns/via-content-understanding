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

def prob_ranked_corectly(scores_a, scores_b, label_a, label_b):
    """Returns the probability that the scores with be ranked correctly.

    This function uses sklearn's AUC function to calculate the probability that
    scores_a and scores_b are ranked appropriately relative to each other.

    The scores vector we input to the AUC function is just the concatenation of
    scores_a and scores_b, while the labels vector we input to the AUC function
    is label_a for the first len(scores_a) items and label_b for the rest of the
    vector. The resulting probability is the return value.

    Args:
        scores_a: a vector of scores with ground truth label label_a
        scores_b: a vector of scores with ground truth label label_b
        label_a: the label for scores in scores_a
        label_b: the label for scores in scores_b
    """
    scores = np.concatenate((scores_a, scores_b))
    labels = np.zeros_like(scores)
    labels[:len(scores_a)] = label_a
    labels[len(scores_a):] = label_b

    probability = metrics.roc_auc_score(labels, scores)

    return probability


def prob_scores_desired_ranked_above_scores_undesired(scores_desired,
    scores_undesired, label_desired, label_undesired):
    """Returns the probability of ranking desired scores over undesired ones.

    This function confirms the labels are valid and then uses area under the
    curve function to compute the probability that items in scores_desired are
    ranked more desired than items in scores_undesired, according to
    label_desired and label_undesired.

    Args:
        scores_desired: a vector of scores with a "desired" label.
        scores_undesired: a vector of scores with an "undesired" label.
        label_desired: the "desired" label. Should be 0 or 1.
        label_undesired: the "undesired" label. Should be 0 or 1.

    Raises:
        ValueError: raised if the labels are the same or not in {0, 1}.

    """
    validate_labels(label_desired, label_undesired)

    probability = prob_ranked_corectly(scores_desired, scores_undesired,
        label_desired, label_undesired)

    return probability
