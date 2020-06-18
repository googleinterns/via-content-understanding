"""Implementation of the video encoder.

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
from models.layers import TemporalAggregationLayer, \
    ExpertProjectionModulationLayer, GatedEmbeddingModule


class VideoEncoder(tf.keras.Model):
    """Implementation of the video encoder.

    Attributes:
        experts: a list of experts (that implement BaseExpert)
        expert_aggregated_size: the dimensionality we'll project experts to.
        encoded_expert_dimensionality: the dimensionality experts embeddings
            are computed down to. Final output size is num of experts *
            encoded_expert_dimensionality.
        temporal_aggregation_layers: a list of temporal aggregation layers, one
            per expert.
        expert_projection: An expert projection modulation layer.
        g_mlp: A standard feedforward deep neural network, with g_mlp_layers.
        h_mlp: A standard feedforward deep neural network, with h_mlp_layers.
        gems: A list of gated embedding modules, one per network.
        activiation_fn:
        use_batch_norm: 
    """

    def __init__(self, 
            experts,
            expert_aggregated_size=768,
            encoded_expert_dimensionality=100,
            g_mlp_layers=5,
            h_mlp_layers=5,
            make_activation_layer=tf.keras.layers.ReLU,
            use_batch_norm=True
            ):
        """Initalize video encoder.

        Parameters:
            experts: a list of experts (that implement BaseExpert)
            expert_aggregated_size: the dimensionality we'll project experts to.
            encoded_expert_dimensionality: the dimensionality experts embeddings
                are computed down to. Final output size is num of experts * 
                encoded_expert_dimensionality.
            g_mlp_layers: layers in the mlp labeled "g".
            h_mlp_layers: layers in the mlp labeled "h".
            a
        """

        super(VideoEncoder, self).__init__()

        self.experts = experts
        self.expert_aggregated_size = expert_aggregated_size
        self.encoded_expert_dimensionality = encoded_expert_dimensionality
        self.make_activation_layer = make_activation_layer
        self.use_batch_norm = use_batch_norm

        self.make_temporal_aggregation_layers()
        self.g_mlp = self.make_mlp(g_mlp_layers)
        self.h_mlp = self.make_mlp(h_mlp_layers)

        self.expert_projection = ExpertProjectionModulationLayer()

        self.make_gem_layers()

    def make_temporal_aggregation_layers(self):
        """Make temporal aggregation layers."""
        self.temporal_aggregation_layers = []

        for expert in self.experts:
            should_use_netvlad = len(expert.embedding_shape) > 1
            
            self.temporal_aggregation_layers.append(TemporalAggregationLayer(
                self.expert_aggregated_size,
                should_use_netvlad
                ))

    def make_mlp(self, num_layers):
        """Makes and returns an mlp with num_layers layers."""
        sequential_layers = []
        
        for _ in range(num_layers):
            if self.use_batch_norm:
                sequential_layers.append(
                    tf.keras.layers.BatchNormalization())

            sequential_layers.append(
                self.make_activation_layer())

            sequential_layers.append(
                tf.keras.layers.Dense(
                    self.expert_aggregated_size,
                    activation=None))

        return tf.keras.Sequential(sequential_layers)

    def make_gem_layers(self):
        """Create gated embedding module layers."""
        self.gems = []
        
        for _ in self.experts:
            self.gems.append(GatedEmbeddingModule(
                self.expert_aggregated_size,
                self.encoded_expert_dimensionality
            ), True)

    def call(self, inputs):
        """Forward pass on the video encoder."""
        assert len(inputs) == len(self.experts)

        aggregated_embeddings = self.temporal_aggreagation(inputs)
        output_embedding = self.collaborative_gating(aggregated_embeddings)

        return output_embedding

    def temporal_aggreagation(self, inputs):
        """Run temporal aggregation module."""
        aggregated_embeddings = []

        for aggregator, embedding in zip(
            self.temporal_aggregation_layers, inputs):
            aggregated_embeddings.append(aggregator(embedding))

        return aggregated_embeddings

    def collaborative_gating(self, aggregated_embeddings):
        """Run collaboartive gating module."""
        gated_embeddings = []

        for i, embedding in enumerate(aggregated_embeddings):
            summed_pairwise_attentions = None

            for j, other_embedding in enumerate(aggregated_embeddings):
                if i != j:
                    
                    attentions = self.g_mlp(tf.concat(
                        [embedding, other_embedding],
                        axis=1,
                    ))

                    if summed_pairwise_attentions is None:
                        summed_pairwise_attentions = attentions
                    else:
                        summed_pairwise_attentions += attentions

            attentions = self.h_mlp(summed_pairwise_attentions)

            embeddings = self.expert_projection([embedding, attentions])

            gated_embedding = self.gems[i](embeddings)

            gated_embeddings.append(gated_embedding)

        return tf.concat(gated_embeddings, axis=1)
