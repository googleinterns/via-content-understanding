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

def bidirectional_max_margin_ranking_loss(video_embeddings, text_embeddings, m):
    """Implementation of the Bidirectional max margin ranking loss.

    Arguments:
        video_embeddings: a tensor of dimension (batch size x embedding space
            size) where the ith element is the embedding for the ith video in
            the batch
        text_embeddings: a tensor of dimension (batch size x embedding space
            size) where the ith element is the embedding for the ith caption in
            the batch
    """

    batch_size = video_embeddings.shape[0]

    video_embeddings = tf.expand_dims(video_embeddings, 1)
    text_embeddings = tf.expand_dims(text_embeddings, 0)

    video_embeddings = tf.tile(video_embeddings, [1, batch_size, 1])
    text_embeddings = tf.tile(text_embeddings, [batch_size, 1, 1])

    similarities = tf.reduce_sum(video_embeddings * text_embeddings, axis=-1)

    diag = tf.linalg.tensor_diag_part(similarities)
    self_similarities = tf.tile(tf.expand_dims(diag, 1), [1, batch_size])

    similarities = similarities + m - self_similarities

    similarities = (1 - tf.eye(batch_size)) * similarities

    similarities = tf.nn.relu(similarities)

    loss = tf.reduce_sum(similarities) / (batch_size**2)

    return loss
