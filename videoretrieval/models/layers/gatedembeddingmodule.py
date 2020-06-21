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

    """
    def __init__(self, input_dimension, output_dimension, include_projection):
        super(GatedEmbeddingModule, self).__init__()

        self.include_projection = include_projection

        if self.include_projection:
            self.linear_layer_one = tf.keras.layers.Dense(
                output_dimension, input_shape=(input_dimension,))

        self.linear_layer_two = tf.keras.layers.Dense(
            output_dimension, input_shape=(output_dimension,))

        self.layer_one_batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_two_batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        if self.include_projection:
            layer_one_activations = self.linear_layer_one(inputs)
        else:
            layer_one_activations = inputs

        layer_two_activations = self.linear_layer_two(layer_one_activations)

        layer_one_activations = self.layer_one_batch_norm(
            layer_one_activations)
        layer_two_activations = tf.math.sigmoid(
            self.layer_two_batch_norm(layer_two_activations))

        unscaled_activations = layer_one_activations * layer_two_activations

        scaled_activations = tf.math.l2_normalize(unscaled_activations, axis=-1)

        return scaled_activations
