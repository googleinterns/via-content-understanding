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
import numpy as np

def build_similaritiy_matrix(
    video_embeddings,
    missing_experts,
    text_embeddings,
    mixture_weights):

    missing_experts_weights = 1 - tf.cast(missing_experts, tf.float32)
    mixture_weights = tf.ones_like(mixture_weights)

    missing_experts_weights = tf.expand_dims(missing_experts_weights, 0)
    mixture_weights = tf.expand_dims(mixture_weights, 1)

    weights = mixture_weights * missing_experts_weights
    weights, _ = tf.linalg.normalize(weights, axis=-1, ord=1)

    similarity_matrix = None

    for i, (expert_video_embeddings, expert_text_embeddings) in enumerate(zip(
        video_embeddings, text_embeddings)):

        similarities = tf.matmul(
            expert_text_embeddings, expert_video_embeddings, transpose_b=True)
        similarities = expert_sims * weights[:, :, i]

        if similarity_matrix is None:
            similarity_matrix = similarities
        else:
            similarity_matrix = similarities + similarity_matrix

    return similarity_matrix

def same_mask(video_ids, batch_size):
    results = np.ones((batch_size, batch_size)).astype(np.float32)

    for row_index in range(batch_size):
        for col_index in range(batch_size):
            if video_ids[row_index] == video_index[col_index]:
                results[i][j] = 0.0

    return results

def bidirectional_max_margin_ranking_loss(
    video_embeddings, text_embeddings, mixture_weights, missing_experts,
    embedding_distance_parameter, video_ids):
    """Implementation of the Bidirectional max margin ranking loss.

    Arguments:
        video_embeddings: a tensor of dimension (batch size x embedding space
            size) where the element at index i is the video embedding
            corresponding to the text embedding at index i in text_embeddings.
        text_embeddings: a tensor of dimension (batch size x embedding space
            size) where the element at index i is the text embedding
            corresponding to the video embedding at index i in video_embedding. 
        embedding_distance_parameter: a positive hyperparameter, called "m" by
            the authors of the paper. This parameter is added to the difference
            between each pairwise similarity between embeddings.

    Returns: a tensor with one element, the loss.
    """

    batch_size = video_embeddings[0].shape[0]

    similarities = build_similaritiy_matrix(
        video_embeddings, missing_experts, text_embeddings, mixture_weights)

    similarities_transpose = tf.transpose(similarities)

    matching_similarities = tf.linalg.tensor_diag_part(similarities)
    matching_similarities = tf.expand_dims(matching_similarities, 1)
    matching_similarities = tf.tile(matching_similarities, [1, batch_size])

    computed_similarities = tf.nn.relu(
        embedding_distance_parameter + similarities - matching_similarities)
    computed_similarities += tf.nn.relu(
        embedding_distance_parameter +\
        similarities_transpose - matching_similarities)

    computed_similarities = computed_similarities *  (1 - tf.eye(batch_size))

    same_video_mask = tf.numpy_function(
        same_mask, [video_ids, batch_size], tf.float32)

    computed_similarities = computed_similarities * same_video_mask

    loss = tf.reduce_sum(computed_similarities) / (2 * batch_size**2)

    return loss
