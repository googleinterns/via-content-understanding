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

def build_similaritiy_matrix(
    video_embeddings,
    missing_experts,
    text_embeddings,
    mixture_weights):

    num_videos = video_embeddings.shape[0]
    num_text = text_embeddings.shape[0]

    batch_index = 0
    num_of_experts_index = 1
    expert_dimensionality_index = 2

    text_embeddings = tf.transpose(
        text_embeddings, [
            num_of_experts_index,
            batch_index,
            expert_dimensionality_index
        ])

    video_embeddings = tf.transpose(
        video_embeddings, [
            num_of_experts_index,
            expert_dimensionality_index,
            batch_index
        ])

    expert_wise_similarity = tf.matmul(text_embeddings, video_embeddings)


    missing_weight_multipliers = 1 - tf.cast(mixture_weights, tf.float32)

    tiled_mixture_weights = tf.tile(
        tf.expand_dims(mixture_weights, 1), [1, num_videos, 1])

    tiled_missing_weights = tf.tile(
        tf.expand_dims(missing_weight_multipliers, 0), [num_text, 1, 1])

    weight_multipliers, _ = tf.normalize(
        tiled_mixture_weights * tiled_missing_weights,
        ord=1, axis=-1)

    similarity_matrix = tf.reduce_sum(
        expert_wise_similarity * weight_multipliers, axis=-1)

    return similarity_matrix

def bidirectional_max_margin_ranking_loss(
    video_embeddings, text_embeddings, mixture_weights, missing_experts,
    embedding_distance_parameter):
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

    batch_size = video_embeddings.shape[0]


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

    loss = tf.reduce_sum(computed_similarities) / (batch_size**2)

    return loss
