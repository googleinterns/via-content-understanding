"""Defines functions for computing ranks.

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

import tensorflow as tf
from .loss import build_similarity_matrix

parallel_iterations = 32

@tf.function
def compute_rank(input_):
    """Compute and returns the position that an item of a tensor is ranked at.

    Arguments: 
        input_: a tuple of two elements. First, a tensor of similarities,
            second, the index of the correct similarity.
    """

    similarities, index = input_
    pair_similarity = similarities[index]
    rank = tf.reduce_sum(tf.cast(similarities >= pair_similarity, tf.int32))
    return rank

@tf.function
def compute_ranks(similarity_matrix):
    """Computes ranks for a batch of video and text embeddings from a similarity
        matrix.

    Arguments:
        similarity_matrix: a batch size x batch size matrix, where the item in
            the ith row and and jth column represents the similarity between the
            ith query and the jth stored value.

    Returns: a tensor of shape batch_size containing the computed rank of each
        query.
    """

    ranks_tensor = tf.map_fn(
        compute_rank,
        (similarity_matrix, tf.range(similarity_matrix.shape[0])), 
        dtype=tf.int32,
        parallel_iterations=parallel_iterations)

    return ranks_tensor

@tf.function
def get_mean_rank(ranks_tensor):
    """Gets the mean rank given a tensor of ranks.

    Parameters:
        ranks_tensor: an integer tensor of ranks.

    Returns:
        The mean rank as a float32 tensor with one element.
    """
    return tf.reduce_mean(tf.cast(ranks_tensor, tf.float32))

@tf.function
def get_median_rank(ranks_tensor):
    """Gets the median rank given a tensor of ranks.

    Parameters:
        ranks_tensor: an integer tensor of ranks.

    Returns:
        The median rank as an int32 tensor with one element.
    """
    return tf.reduce_min(
        tf.math.top_k(
            ranks_tensor, ranks_tensor.shape[0] // 2, sorted=False)[0])

def get_recall_at_k(ranks_tensor, k):
    """Gets the recall at k given a tensor of ranks and a threshold k.

    Parameters:
        ranks_tensor: an integer tensor of ranks.
        k: the threshold used in the recall calculation.

    Returns:
        The recall at k as a float32 tensor with one element.
    """
    recalled_correctly_mask = tf.cast(ranks_tensor <= k, tf.float32)
    return tf.reduce_mean(recalled_correctly_mask)
