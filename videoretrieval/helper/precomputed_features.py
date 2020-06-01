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

Functions to download, extract, and process precomputed features.
"""

from helper import file_downloader, tar_helper
from cache import features_cache
import pickle

def download_features_tar(features_tar_url, features_tar_path):
    """Downloads a tar file of precomputed features.

    Arguments:
        features_tar_url: a string that is the url of the features tar.
        features_tar_path: the path the tar should be saved to.
    """

    features_tar_path.parent.mkdir(parents=True, exist_ok=True)

    file_downloader.download_by_url(
        features_tar_url,
        features_tar_path
    )

def extract_features_from_file(file):
    """Extracts features from a file using `pickle.load`."""
    return pickle.load(file)

def verify_features_dimensioality(features, expert):
    """Checks that the dimensionality of the embeddings matches expectations."""

    feature_shape = next(iter(list(features.values()))).shape

    if expert.embedding_shape != feature_shape:
        print(f"{expert.name}: {expert.embedding_shape} | {feature_shape}")

def map_features_to_dict(features, dataset, expert):
    """Returns features as a dictionary."""

    if type(features) != dict:
        try:
            features = features.todict()
        except AttributeError:
            raise NotImplementedError(
                f"Type of cached features for {dataset.dataset_name}" + \
                "{expert.name} unknown")

    return features

def cache_features(dataset, expert_to_features, features_tar_path):
    """Caches features for a given dataset and tar of features.

    Arguments:
        dataset: a BaseDataset class for the dataset that the embeddings are
            for.
        expert_to_features: a dict that maps from experts to the path (inside
            the tar) of the file with the embeddings.
        features_tar_path: the path to the tar with the embeddings.
    """

    experts = []
    paths = []

    for expert, path in expert_to_features.items():
        experts.append(expert)
        paths.append(path)

    features_file_generator = tar_helper.generate_files_from_tar(
        features_tar_path, paths)


    for expert, features_file in zip(experts, iter(features_file_generator)):
        features = extract_features_from_file(features_file)

        features = map_features_to_dict(features, dataset, expert)
        verify_features_dimensioality(features, expert)

        features_cache.cache_features_by_expert_and_dataset(
            dataset, expert, features)
        
def download_and_cache_precomputed_features(dataset, features_tar_url,
    features_tar_path, expert_to_features):
    """Downloads and caches precomputed features.

    Arguments:
        dataset: a BaseDataset class for the dataset that the embeddings are
            for.
        features_tar_url: the url of the features tar.
        features_tar_path: the path to download the tar to.
        expert_to_features: a dict that maps from experts to the path (inside
            the tar) of the file with the embeddings.
    """

    download_features_tar(features_tar_url, features_tar_path)
    cache_features(dataset, expert_to_features, features_tar_path)