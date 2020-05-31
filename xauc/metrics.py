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
import probability

class xAUCMetrics:
    """A wrapper class for xauc, xauc0, and xauc1 metrics."""

    def __init__(self, xauc, xauc0, xauc1):
        """Initalize the xAUC metrics class."""
        self.xauc = xauc
        self.xauc0 = xauc0
        self.xauc1 = xauc1

class ClassScores:
    """A wrapper class for wrapping preferred/undesired scores of one class."""

    def __init__(self, scores_preferred_outcome, scores_undesired_outcome):
        """Initalize the ClassScores wrapper class."""
        self.preferred = scores_preferred_outcome
        self.undesired = scores_undesired_outcome 

def compute_class_xauc(
    probability_calculator
    class_scores
    other_scores,
    all_scores):
    """Compute and return the xauc metrics for a given class.

    Arguments:
        probability_calculator: an instance of the ProbabilityCalculator class
            used to calculate the metrics, initalized with the labels that the
            data has.
        class_scores: an instance of the ClassScores class

    Returns: an instance of xAUCMetrics that contains the xAUC, xAUC 0 and 
        xAUC 1 metrics for the given class.
    """    
    xauc = probability_calculator.probability_preferred_ranked_above_undesired(
        class_scores.preferred, other_scores.undesired)

    xauc0 = probability_calculator.probability_preferred_ranked_above_undesired(
        all_scores.preferred, class_scores.undesired)

    xauc1 = probability_calculator.probability_preferred_ranked_above_undesired(
        class_scores.preferred, all_scores)

    return xAUCMetrics(xauc=xauc, xauc0=xauc, xauc1=xauc1)


def calculate_xauc_metrics(
    protected_class_scores_preferred_outcome,
    protected_class_scores_undesired_outcome,
    other_class_scores_preferred_outcome,
    other_class_scores_undesired_outcome,
    preferred_outcome_label,
    undesired_label):

    probability_calculator = probability.ProbabilityCalculator(
        preferred_outcome_label, undesired_label)

    all_scores_preferred_outcome = \
        protected_class_scores_preferred_outcome +\
        other_class_scores_preferred_outcome

    all_scores_undesired_outcome = \
        protected_class_scores_undesired_outcome +\
        other_class_scores_undesired_outcome

    protected_class_scores = ClassScores(
        protected_class_scores_preferred_outcome,
        protected_class_scores_undesired_outcome)

    other_class_scores = ClassScores(
        other_class_scores_preferred_outcome, 
        other_class_scores_undesired_outcome)

    all_scores = ClassScores(
        all_scores_preferred_outcome,
        all_scores_undesired_outcome)

    protected_class_xauc_scores = compute_class_xauc(
        probability_calculator,
        protected_class_scores,
        other_class_scores,
        all_scores)

    other_class_xauc_scores = compute_class_xauc(
        probability_calculator,
        other_class_scores,
        protected_class_scores,
        all_scores)

    return protected_class_xauc_scores, other_class_xauc_scores
