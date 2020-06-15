# Lint as: python3
"""Helper for validating that labels are an acceptable value.

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

def validate_labels(desired_label, undesired_label):
    """Verifies validity of labels.

    Confirms that the labels are not the same and both in {0, 1}.

    Args:
        label_desired: the label for examples that are "desired". Either 0 or 1.
        label_undesired: the label for examples that are "undesired". Either 0 
            or 1.

    Returns: nothing

    Raises:
        ValueError: raised if the labels are the same or not in {0, 1}.
    """

    if desired_label != 0 and desired_label != 1:
        raise ValueError("desired label is not a legal label")

    if undesired_label != 0 and undesired_label != 1:
        raise ValueError("undesired label is not a legal label")

    if desired_label == undesired_label:
        raise ValueError("undesired label cannot be equal to desired label")
