"""Implementation of the text encoder.

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

from models.layers import GatedEmbeddingModule, NetVLAD

class TextEncoder(tf.keras.Model):
    """Implementation of Text Encoder.

    This model takes in contextual embeddings from a language model and maps
    them to a fixed length, sharded embedding that corresponds to embeddings
    produced by the corresponding video encoder. This model should be trained
    in concert with a video encoder.

    The contextual embeddings are first aggregated using netvlad to a fixed
    length vector. Then, for each expert in the corresponding video encoder,
    this fixed length vector is passed through a Gated Embedding Module to
    produce a shard of an embedding. The fixed length vector is also passed
    through a single dense layer to produce weights that specify the relative
    importance of the shards of the embeddings.

    Attributes:
        num_of_experts: the number of experts used.
        num_netvlad_clusters: the number of clusters in the NetVLAD model.
        language_model_dimensionality: the length of the last dimension of
            the contextual embeddings.
        netvlad: a NetVLAD model, used for aggregation embeddings.
        encoded_expert_dimensionality: dimensionality of each expert in the
            final embedding.
        gems: a list of gated embedding modules, one per expert.
        dense_layers: a list of dense layers, one per expert.

    """
    def __init__(self,
            num_of_experts,
            encoded_expert_dimensionality=768,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros"):
        """Initialize this model.

        Parameters:
            num_of_experts: number of experts used in the video encoder.
            num_netvlad_clusters: number of clusters in NetVLAD.
            ghost_clusters: number of ghost clusters in NetVLAD.
            language_model_dimensionality: last dimension of output of language
                model.
            encoded_expert_dimensionality: the dimensionality video experts
                embeddings are computed down to.
            kernel_initializer: the strategy used to initialize the weights in
                dense layers' kernel.
            bias_initial: the strategy used to initialize the weights in dense
                layers' biases.
        """
        super(TextEncoder, self).__init__()

        self.num_of_experts = num_of_experts
        self.encoded_expert_dimensionality = encoded_expert_dimensionality

        self.make_gems(
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)

        self.make_dense_layers(
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)

    def make_gems(self, kernel_initializer, bias_initializer):
        """Initialize gated embedding modules."""
        self.gems = []

        for _ in range(self.num_of_experts):
            self.gems.append(GatedEmbeddingModule(
                self.encoded_expert_dimensionality,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer))


    def make_dense_layers(self, kernel_initializer, bias_initializer):
        """Make dense layer used for generating mixture of embedding weights.
        Note: "moe" stands for mixture of embeddings weights. 
        """

        self.moe_dense = tf.keras.layers.Dense(
            self.num_of_experts,
            activation="softmax",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)

    def call(self, input_):
        """Executes a forward pass on the text encoder.

        First, the text is aggregated using netvlad. These aggregated
        text embeddings are inputted to each gated embedding module to generate 
        the normalized embeddings. The aggregated text embeddings are also
        inputted into a dense layer to generate the mixture weights.

        Parameters:
            input_: a batch of contextual embeddings.

        Returns: a tuple of two elements. First, a list of embeddings for the
        text captions. Each element of this list is a tensor of shape batch size
        x encoded expert dimensionality. Second, a tensor of mixture weights
        for the embeddings of shape batch size x number of experts.
        """

        expert_embeddings = []

        for expert_gated_embedding_module in self.gems:
            expert_embedding = expert_gated_embedding_module(
                input_)

            expert_embeddings.append(expert_embedding)

        mixture_weights = self.moe_dense(aggregated_embeddings)

        return expert_embeddings, mixture_weights
