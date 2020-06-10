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
import tensorflow as tf

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
        """Returns a dict of that maps from video id to a list of captions."""
        pass

    @property
    @abstractmethod
    def train_valid_test_ids(self):
        """Returns a tuple of sets providing ids for the dataset splits.

        Returns: a tuple of sets, where the first set contains the ids for the 
        train data, the second for the validation data, and the third for the
        test data."""

        pass

    def build_generator(self, data):
        """Build a generator that yields each element from data."""
        for example in data:
            yield example

    def build_id_caption_pair_generator_dataset(self, data):
        """Build a tf.data Dataset out of id caption pairs in data."""
        generator = lambda: self.build_generator(data)

        return tf.data.Dataset.from_generator(generator, (tf.string, tf.string))
    
    @property
    def id_caption_pair_datasets(self):
        """Gets id caption pair datasets for each split in dataset.

        Returns: a tuple of three tuples, where the first element of each tuple
        is the tf.data.Dataset of video id caption pairs, and the second element
        is the name of the split as a string. In the retured tuple, the first
        element is the data for the train split, followed by the valid and test
        sets.
        """ 

        train_ids, valid_ids, test_ids = self.train_valid_test_ids

        train_data = []
        valid_data = []
        test_data = []

        for video_id, caption in self.video_captions:
            if video_id in train_ids:
                train_data.append((video_id, caption))
            elif video_id in valid_ids:
                valid_data.append((video_id, caption))
            elif video_id in test_ids:
                test_data.append((video_id, caption))
            else:
                print(f"Orphan pair: id: {video_id}, caption: {hash(caption)}")

        self.num_of_train_examples = len(train_data)
        self.num_of_valid_examples = len(valid_data)
        self.num_of_test_examples = len(test_data)

        train_dataset = self.build_id_caption_pair_generator_dataset(train_data)
        valid_dataset = self.build_id_caption_pair_generator_dataset(valid_data)
        test_dataset = self.build_id_caption_pair_generator_dataset(test_data)

        return (
            (train_dataset, "train"), 
            (valid_dataset, "valid"), 
            (test_dataset, "test")
        )

    def num_of_examples_by_split(self, split_name):
        """Get the number of examples in the given split in this dataset."""
        if "num_of_train_examples" not in dir(self):
            _ = self.id_caption_pair_datasets

        if split_name == "train":
            return self.num_of_train_examples
        elif split_name == "valid":
            return self.num_of_valid_examples
        elif split_name == "test":
            return self.num_of_test_examples
        else:
            raise ValueError("Illegal split name")
