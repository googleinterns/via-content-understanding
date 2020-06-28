"""Implementation of the Gated Embedding Module.

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

class GatedEmbeddingModule(tf.keras.layers.Layer):
    """An implementation of the gating embeding module.

    The gated embedding module takes one tensor (Z_0) of dimension
    input_dimension as input and returns a tensor of output_dimension.

    The equations that define the gated embedding module are as follows.

    Z_1 = W_1 * Z_0 + B_1
    Z_2 = multiply(Z_1, Sigmoid(W_2 * Z_1 + B_2))
    Output = Z_2 / norm(Z_2)

    There are four trainable variables, W_1, B_1, W_2, and B_2, which we can
    represent in linear_layer_one and linear_layer_two.

    Attributes:
        linear_layer_one: a Dense layer that performs W_1 * Z_0 + B_1.
        linear_layer_two: a Dense layer that performs W_2 * Z_1 + B_2 and
            applies a sigmoid.
        layer_one_batch_norm: a Batch Normalization layer for the activiations
            from layer one.
        layer_two_batch_norm: a Batch Normalization layer for the activations
            from layer two.
    """
    def __init__(self, input_dimension, output_dimension):
        super(GatedEmbeddingModule, self).__init__()

        self.linear_layer_one = tf.keras.layers.Dense(
            output_dimension, input_shape=(input_dimension,))

        self.linear_layer_two = tf.keras.layers.Dense(
            output_dimension, input_shape=(output_dimension,))

        self.batch_norm = tf.keras.layers.BatchNormalization(momentum=0.1)

    def call(self, inputs):
        layer_one_activations = self.linear_layer_one(inputs)
        layer_two_activations = self.linear_layer_two(layer_one_activations)

        layer_two_activations = self.batch_norm(layer_two_activations)
        layer_two_activations = tf.nn.sigmoid(layer_two_activations)

        unscaled_activations = layer_one_activations * layer_two_activations

        scaled_activations = tf.math.l2_normalize(unscaled_activations, axis=-1)

        return scaled_activations

class GatedEmbeddingUnitReasoning(tf.keras.layers.Layer):
    def __init__(self, output_dimension):
        self.fully_connected = tf.keras.layers.Dense(output_dimension)

        self.batch_norm_one = tf.keras.layers.BatchNormalization(momentum=0.1)
        self.batch_norm_two = tf.keras.layers.BatchNormalization(momentum=0.1)

    def call(self, inputs):
        assert len(inputs) == 2

        embedding, mask = inputs

        activations = self.fully_connected(activations)

        activations = self.batch_norm_one(activations)
        mask = self.batch_norm_two(mask)

        output = embedding * tf.nn.sigmoid(activations + mask)

        return tf.math.l2_normalize(output, axis=-1)