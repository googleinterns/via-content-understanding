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
Defines the loss function needed to train the model.
"""

import tensorflow as tf
from .loss import build_similaritiy_matrix

@tf.function
def compute_rank(similarities, index):
    pair_similarity = similarities[index]
    rank = tf.reduce_sum(tf.cast(similarities >= pair_similarity, tf.int32))
    return rank

@tf.function
def compute_ranks(
    query_embeddings, mixture_weights, static_embeddings, missing_experts):

    similarty_matrix = build_similaritiy_matrix(
        static_embeddings, missing_experts, query_embeddings, mixture_weights)

    ranks_tensor = tf.map_fn(
        (compute_rank, tf.range(similarty_matrix.shape[0])), 
        dtype=tf.int32,
        parallel_iterations=8)

    return ranks_tensor

@tf.function
def get_mean_rank(ranks_tensor):
    return tf.reduce_mean(ranks_tensor)

@tf.function
def get_median_rank(ranks_tensor):
    return tf.reduce_min(
        tf.math.top_k(ranks_tensor, ranks_tensor.shape[0] // 2, sorted=False))

@tf.function
def get_recall_at_k(ranks_tensor, k):
    recalled_correctly_mask = tf.cast(ranks_tensor <= k, tf.float32)
    return tf.reduce_mean(recalled_correctly_mask)
