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

    Attributes:
        num_of_experts: the number of experts used.
        num_netvlad_clusters: the number of clusters in the NetVLAD model.
        language_model_dimensionality: last dimension of output of language
            model.
        netvlad: A NetVLAD model, used for aggregation embeddings.
        encoded_expert_dimensionality: dimensionality of each expert in the
            final embedding.
        gems: A list of gated embedding modules, one per expert.
        dense_layers: A list of dense layers, one per expert.

    """
    def __init__(self,
            num_of_experts,
            num_netvlad_clusters=25,
            ghost_clusters=1,
            language_model_dimensionality=768,
            encoded_expert_dimensionality=100):
        """Initialize TextEncoder.

        Parameters:
            num_of_experts: number of experts used in the video encoder.
            num_netvlad_clusters: number of clusters
            language_model_dimensionality: last dimension of output of language
                model.
            encoded_expert_dimensionality: the dimensionality video experts
                embeddings are computed down to. Final output size is num of
                experts * encoded_expert_dimensionality.
        """
        super(TextEncoder, self).__init__()

        self.num_of_experts = num_of_experts
        self.num_netvlad_clusters = num_netvlad_clusters
        self.language_model_dimensionality = language_model_dimensionality
        self.netvlad = NetVLAD(num_netvlad_clusters, ghost_clusters)
        self.encoded_expert_dimensionality = encoded_expert_dimensionality

        self.make_gems()
        self.make_dense_layers()

    def make_gems(self):
        """Initialize gated embedding modules."""
        self.gems = []

        for _ in range(self.num_of_experts):
            self.gems.append(GatedEmbeddingModule(
                self.language_model_dimensionality * self.num_netvlad_clusters,
                self.encoded_expert_dimensionality))


    def make_dense_layers(self):
        """Make dense layer used for generating mixture of embedding weights.
        Note: "moe" stands for mixture of embeddings weights. 
        """

        self.moe_dense = tf.keras.layers.Dense(
            self.num_of_experts,
            activation="softmax")

    def call(self, input_):
        """Executes a forward pass on the text encoder.

        First, the text is aggregated using netvlad. These aggregated
        text embeddings are inputted to each gated embedding module to generate 
        the normalized embeddings. The aggregated text embeddings are also
        inputted into the moe_dense layer to generate the mixture weights.

        Returns: a tuple of two elements. First, a list of embeddings for the
        text captions. Second, a tensor of mixture weights for the embeddings.

        """

        aggregated_embeddings = self.netvlad(input_)

        expert_embeddings = []

        for expert_gated_embedding_module in self.gems:
            expert_embedding = expert_gated_embedding_module(
                aggregated_embeddings)

            expert_embeddings.append(expert_embedding)

        mixture_weights = self.moe_dense(aggregated_embeddings)

        return expert_embeddings, mixture_weights
