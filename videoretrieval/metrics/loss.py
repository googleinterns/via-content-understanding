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

def build_similarity_matrix(
    video_embeddings,
    text_embeddings,
    mixture_weights,
    missing_experts):
    """Builds a similarity matrix between text_embeddings and video_embeddings.

    Let m be the batch size, n be the number of experts, d be the embedding
    dimensionality. 

    Args:
        video_embeddings: a list of length n, where the ith element is the
            video embedding for the ith expert as a tensor of shape m x d.
        text_embeddings: a list of length n, where the ith element is the
            text embedding for the ith expert as a tensor of shape m x d.
        mixture_weights: a tensor of shape m x n, containing mixture weights for
            a corresponding text embedding.
        missing_experts: a boolean tensor of shape b x m, where each element
            corresponds to a video embedding and indicates the missing experts. 
        
    Returns: A batch_size x batch_size tensor, where the value in the ith row
        and jth column is the similarity between the ith text embedding and the
        jth video embedding. 
    """
    assert len(text_embeddings) > 0
    assert len(video_embeddings) > 0

    num_text_embeddings = text_embeddings[0].shape[0]
    num_video_embeddings = video_embeddings[0].shape[0]

    missing_experts_weights = 1 - tf.cast(missing_experts, tf.float32)

    # Expand dimensions to allow for multiplication.
    missing_experts_weights = tf.expand_dims(missing_experts_weights, 0)
    mixture_weights = tf.expand_dims(mixture_weights, 1)

    weights = mixture_weights * missing_experts_weights
    weights, _ = tf.linalg.normalize(weights, axis=-1, ord=1)

    similarity_matrix = tf.zeros(
        (num_text_embeddings, num_video_embeddings), tf.float32)

    for expert_index, (
        expert_video_embeddings, expert_text_embeddings) in enumerate(
            zip(video_embeddings, text_embeddings)):

        similarities = tf.matmul(
            expert_text_embeddings, expert_video_embeddings, transpose_b=True)
        similarities = similarities * weights[:, :, expert_index]

        similarity_matrix = similarities + similarity_matrix

    return similarity_matrix

def bidirectional_max_margin_ranking_loss(
    similarity_matrix, embedding_margin):
    """Implementation of the Bidirectional max margin ranking loss.

    Args:
        similarity_matrix: a batch size x batch size similarity matrix, where
            the element in the ith row and jth column is the similarity between
            the ith text embedding and the jth video embedding.
        embedding_margin: a positive margin hyper-parameter, called
            "m" by the authors of the paper. This parameter is added to the
            difference between each pairwise similarity between embeddings.

    Returns: A tensor with one element, the loss.
    """

    batch_size = similarity_matrix[0].shape[0]

    similarities_transpose = tf.transpose(similarity_matrix)

    matching_similarities = tf.linalg.tensor_diag_part(similarity_matrix)
    matching_similarities = tf.expand_dims(matching_similarities, 1)
    matching_similarities = tf.tile(matching_similarities, [1, batch_size])

    computed_similarities = tf.nn.relu(
        embedding_margin + similarity_matrix - matching_similarities)
    computed_similarities += tf.nn.relu(
        embedding_margin + similarities_transpose - matching_similarities)

    computed_similarities = computed_similarities *  (1 - tf.eye(batch_size))

    loss = tf.reduce_sum(
        computed_similarities) / (2 * (batch_size**2 - batch_size))

    return loss
