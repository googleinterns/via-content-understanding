""" Copyright 2020 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Defines a base class for datasets.
"""

from abc import ABC as AbstractClass
from abc import abstractmethod
import pathlib

class BaseVideoDataset(AbstractClass):
    """Base class for video datasets."""

    @property
    @abstractmethod
    def dataset_name(self):
        """Gets the name of the dataset."""
        pass    

    @property
    @abstractmethod
    def dataset_downloaded(self):
        """A boolean describing if the dataset is downloaded."""
        pass

    @property
    def data(self):
        """Gets a tf.data object representing the dataset"""
        pass

    @abstractmethod
    def download_dataset(self):
        """Downloads the dataset."""
        pass

    @abstractmethod
    def download_and_cache_precomputed_features(self):
        """Downloads and caches precomputed features for the given dataset."""
        pass

    @property
    @abstractmethod
    def video_captions():
        """Returns a dict of that maps from video_id to a list of captions."""
        pass

    @property
    @abstractmethod
    def train_valid_test_ids(self):
        """Returns a tuple of sets providing ids for the dataset splits.

        Returns: a tuple of sets, where the first set contains the ids for the 
        train data, the second for the validation data, and the third for the
        test data."""

        pass
    