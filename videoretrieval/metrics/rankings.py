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
from .loss import build_similaritiy_matrix

parallel_iterations = 8

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
def compute_ranks(
    text_embeddings, mixture_weights, video_embeddings, missing_experts):
    """Computes a ranks for a batch of video and text embeddings.

    Arguments:
        text_embeddings: a list of text embedding tensors, where each element of
            the list is of shape batch_size x embedding dimensionality.
        mixture_weights: a tensor of mixture weights of shape batch_size x
            number of experts, where each element contains the mixture weights
            for the corresponding text embedding. 
        video_embeddings: a list of video embedding tensors, where each element
            of the list is of shape batch_size x embedding dimensionality.
        missing_experts: a boolean tensor of shape batch_size x number of
            experts, where each element corresponds to a video embedding and
            indicates the missing experts. 

    Returns: a tensor of shape batch_size containg the rank of each element in
        the batch. 
    """

    similarity_matrix = build_similaritiy_matrix(
        video_embeddings, missing_experts, text_embeddings, mixture_weights)

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

@tf.function
def get_recall_at_k(ranks_tensor, k):
    """Gets the recall at k given a tensor of ranks and a k.

    Parameters:
        ranks_tensor: an integer tensor of ranks.
        k: the threshold used in the recall calculation.

    Returns:
        The recall at k as a float32 tensor with one element.
    """
    recalled_correctly_mask = tf.cast(ranks_tensor <= k, tf.float32)
    return tf.reduce_mean(recalled_correctly_mask)
