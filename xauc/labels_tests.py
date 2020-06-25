# Lint as: python3
"""Tests for labels.py.

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
import labels

class TestVerifyLabels(unittest.TestCase):
    """Test cases for the verify labels function."""

    def test_valid_labels(self):
        """Tests the function validate_labels with valid values."""
        try:
            labels.validate_labels(0, 1)
        except ValueError:
            self.fail("test_default_labels failed with labels 0, 1")

        try:
            labels.validate_labels(0.0, 1.0)
        except ValueError:
            self.fail("test_default_labels failed with labels 0.0, 1.0")

        try:
            labels.validate_labels(1, 0)
        except ValueError:
            self.fail("test_default_labels failed with labels 1, 0")

        try:
            labels.validate_labels(0.0, 1.0)
        except ValueError:
            self.fail("test_default_labels failed with labels 0.0, 1.0")

    def test_same_labels(self):
        """Tests the function validate_labels with duplicate labels."""
        with self.assertRaises(ValueError):
            labels.validate_labels(1, 1)

        with self.assertRaises(ValueError):
            labels.validate_labels(1.0, 1.0)

        with self.assertRaises(ValueError):
            labels.validate_labels(0, 0)

        with self.assertRaises(ValueError):
            labels.validate_labels(0.0, 0.0)

    def test_invalid_labels(self):
        """Tests the function test_invalid_labels with invalid labels."""
        with self.assertRaises(ValueError):
            labels.validate_labels(-1, 2)

        with self.assertRaises(ValueError):
            labels.validate_labels(0.5, 2.5)

        with self.assertRaises(ValueError):
            labels.validate_labels("1", "0")

        with self.assertRaises(ValueError):
            labels.validate_labels(0.5, 0.5)

if __name__ == "__main__":
    unittest.main()
