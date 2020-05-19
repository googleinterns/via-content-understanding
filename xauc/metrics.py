# Lint as: python3
"""Implementations of the xAUC metrics.

Implementations of the Cross-Area Under the Curve metric and its balanced
variants, xAUC_0 and xAUC_1.

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

import probability


def x_auc(scores_a_desired, scores_b_undesired, desired_label, undesired_label):
    """Returns the Cross-Area Under Curve (xAUC) metric of the given scores.

    The xAUC metric is defined as the probability "desired" examples in class a
    (the vector scores_a_desired) are ranked "more desired" than "undesired"
    examples in class b (the vector scores_b_undesired). This function uses the
    functions in probability.py to compute the xAUC metric.

    Args:
        scores_a_desired: a vector of scores for examples in class a that have a
            desired ground truth label.
        scores_b_undesired: a vector of scores for examples in class b that have
          an undesired ground truth label.
        desired_label: labels for examples that are "desired". Should be 0 or 1.
        undesired_label: labels for examples that are "undesired". Should be 0
            or 1.

    Raises:
        ValueError: raised if the labels are the same or not in {0, 1}.
    """

    return probability.prob_scores_desired_ranked_above_scores_undesired(
        scores_a_desired, scores_b_undesired, desired_label, undesired_label)


def x_auc_1(scores_g_desired, scores_all_classes_undesired, desired_label,
    undesired_label):
    """Returns the xAUC1 metric of the given scores.

    The xAUC1 metric is defined as the probability "desired" examples in an 
    arbitrary class g (the vector scores_g_desired) are ranked "more desired"
    than "undesired" examples in any class (the vector
    scores_all_classes_undesired). This function uses the functions in
    probability.py to compute the xAUC1 metric.

    Args:
        scores_g_desired: a vector of scores for examples in an arbitrary class
            that have a desired ground truth label.
        scores_all_classes_undesired: a vector of scores for examples of all
            classes with an undesired ground truth label.
        desired_label: the label for examples that are "desired". Should be 0 or
            1.
        undesired_label: the label for examples that are "undesired". Should be
            0 or 1.

    Raises:
        ValueError: raised if the labels are the same, not in {0, 1}.
    """

    return probability.prob_scores_desired_ranked_above_scores_undesired(
        scores_g_desired, scores_all_classes_undesired, desired_label,
        undesired_label)


def x_auc_0(scores_all_classes_desired, scores_g_undesired, desired_label,
    undesired_label):
    """Returns the xAUC0 metric of the given scores.

    The xAUC0 metric is defined as the probability "desired" examples in any
    class (the vector scores_all_classes_desired) are ranked "more desired" than
    "undesired" examples in an arbitrary class g (the vector
    scores_g_undesired). This function uses the functions in probability.py to
    compute the xAUC0 metric.

    Args:
        scores_all_classes_desired: a vector of scores for examples of all
            classes with a desired ground truth label.
        scores_g_undesired: a vector of scores for examples in an 
            arbitrary class that have an undesired ground truth label.
        desired_label: the label for examples that are "desired". Should be 0 or
            1.
        undesired_label: the label for examples that are "undesired". Should be
            0 or 1.

    Raises:
        ValueError: raised if the labels are the same, not in {0, 1}.
    """

    return probability.prob_scores_desired_ranked_above_scores_undesired(
        scores_all_classes_desired, scores_g_undesired, desired_label, 
        undesired_label)
