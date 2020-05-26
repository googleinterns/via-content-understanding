"""Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Defines class to download/load/preprocess MSRVTT dataset.
"""

from base import BaseVideoDataset

from . import metadata
from helper import precomputed_features
from . import constants

import cache

import os

class MSRVTTDataset(BaseVideoDataset):
    """An implementation of BaseVideoDataset for the MSR-VTT dataset."""

    @property
    def dataset_name(self):
        return "msr_vtt"

    @property
    def dataset_downloaded(self):
        return False

    def download_dataset(self):
        """Downloads the dataset and stores it to disk."""
        if self.dataset_downloaded:
            return

        dataset_metadata = metadata.download_and_load_metadata()

    def download_and_cache_precomputed_features(self):
        """Downloads and caches precomputed features."""

        precomputed_features.download_and_cache_precomputed_features(
            self, constants.features_tar_url, constants.features_tar_path,
            constants.expert_to_features
        )
