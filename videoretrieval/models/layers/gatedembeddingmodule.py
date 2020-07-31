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
    def __init__(self, input_dimension, output_dimension):
        super(GatedEmbeddingModule, self).__init__()

        self.linear_layer_one = tf.keras.layers.Dense(
            output_dimension, input_shape=(input_dimension,))
        self.linear_layer_two = tf.keras.layers.Dense(
            output_dimension, input_shape=(output_dimension,), 
            activation="sigmoid")

    def call(self, inputs):
        layer_one_activations = self.linear_layer_one(inputs)
        layer_two_activations = self.linear_layer_two(layer_one_activations)

        unscaled_activations = layer_one_activations * layer_two_activations

        scaled_activations = tf.keras.backend.l2_normalize(unscaled_activations)

        return scaled_activations

class GatedEmbeddingUnitReasoning(tf.keras.layers.Layer):
    """An implementation of the Gated Embedding Unit for Video Reasoning.

    This layer takes in two inputs, an expert video embedding and a mask.
    First, the embedding is passed through a dense layer to create
    activations which are then batch normalized. Then, the mask is batch
    normalized and added to the activations and a sigmoid function is
    applied to the sum, then multiplied element wise with the embedding.
    The this product is then l2 normalized and returned. 

    Attributes:
        fully_connected: a dense layer used to generate activations.
        batch_norm_one: a batch normalization layer for the activations from the
            fully connected layer.
        batch_norm_two: a batch normalization layer for the mask.
    """

    def __init__(self, output_dimension, kernel_initializer, bias_initializer):
        """Initalizes the Gated Embedding Reasoning Unit.

        Arguments:
            output_dimension: dimension this unit should output.
            kernel_initializer: the way to initialize the dense layer's kernels.
            bias_initializer: the way to initialize the dense layer's biases.
        """
        super(GatedEmbeddingUnitReasoning, self).__init__()
        self.fully_connected = tf.keras.layers.Dense(
            output_dimension,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)

        self.batch_norm_one = tf.keras.layers.BatchNormalization(momentum=0.1)
        self.batch_norm_two = tf.keras.layers.BatchNormalization(momentum=0.1)

    def call(self, inputs):
        """Executes a forward pass on this layer.

        Parameters:
            inputs: a pair of two tensors, the first being a video embedding,
            the second being a mask.

        Returns: a tensor, l2 noramlized on the last dimension, of shape batch
            size x output size.
        """
        assert len(inputs) == 2

        embedding, mask = inputs

        activations = self.batch_norm_one(self.fully_connected(embedding))
        mask = self.batch_norm_two(mask)

        output = embedding * tf.nn.sigmoid(activations + mask)

        return tf.math.l2_normalize(output, axis=-1)
