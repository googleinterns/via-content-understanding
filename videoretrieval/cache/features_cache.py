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

Module for caching precomputed features.
"""

from pathlib import Path
import pickle


base_features_path = Path("./downloaded_data/features_cache/")


def get_cache_path(dataset, expert):
    """Gets the file path to cache embeddings from an expert.

    Arguments:
        dataset: a BaseDataset class for the dataset that the embeddings are
            for.
        expert: a BaseExpert class for the expert that the embeddings are from.

    Returns:
        A pathlib object that specifies the file location where the cache will
            be placed.    
    """

    path_directory = base_features_path / dataset.dataset_name

    path_directory.mkdir(parents=True, exist_ok=True)

    return path_directory / (expert.name + ".pkl")


def cache_features_by_expert_and_dataset(dataset, expert, features):
    """Caches the given feature embeddings for the dataset/expert.

    Arguments:
        dataset: a BaseDataset class for the dataset that the embeddings are
            for.
        expert: a BaseExpert calss for the expert that the embeddings are from.
        features: the computed features/embeddings to be cached.

    Returns: None
    """

    file_path = get_cache_path(dataset, expert)

    with open(file_path, "wb") as file:
        pickle.dump(features, file)


def get_cached_features_by_expert_and_dataset(dataset, expert):
    """Returns the cached feature embeddings for the dataset/expert.

    Arguments:
        dataset: a BaseDataset class for the dataset that the embeddings are
            for.
        expert: a BaseExpert class for the expert that the embeddings are from.
    
    Returns: The previously cached features for the given dataset/expert.
        Returns None if there is no value cached.
    """

    file_path = get_cache_path(dataset, expert)

    if not file_path.exists():
        return None

    with open(file_path, "rb") as file:
        cached_features = picle.load(file)

    return cached_features
