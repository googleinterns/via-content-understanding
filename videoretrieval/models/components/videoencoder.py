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
    ExpertProjectionModulationLayer, GatedEmbeddingUnitReasoning


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
        gems: A list of gated embedding modules, one per embedding.
        activation_layer: the type of activation to be used.
        use_batch_norm: if we use batch normalization in the network
    """

    def __init__(self, 
            experts,
            experts_use_netvlad,
            experts_netvlad_shape,
            expert_aggregated_size=768,
            encoded_expert_dimensionality=100,
            g_mlp_layers=5,
            h_mlp_layers=5,
            make_activation_layer=tf.keras.layers.ReLU,
            use_batch_norm=True,
            remove_missing_modalities=True,
            include_self=True
            ):
        """Initalize video encoder.

        Parameters:
            experts: a list of experts (that implement BaseExpert).
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
        self.experts_use_netvlad = experts_use_netvlad
        self.experts_netvlad_shape = experts_netvlad_shape
        self.expert_aggregated_size = expert_aggregated_size
        self.encoded_expert_dimensionality = encoded_expert_dimensionality
        self.make_activation_layer = make_activation_layer
        self.use_batch_norm = use_batch_norm

        self.make_temporal_aggregation_layers()
        self.g_mlp = self.make_mlp(g_mlp_layers)
        self.h_mlp = self.make_mlp(h_mlp_layers)

        self.expert_projection = ExpertProjectionModulationLayer()

        self.make_gem_layers()
        self.remove_missing_modalities = remove_missing_modalities
        self.include_self = include_self

    def make_temporal_aggregation_layers(self):
        """Make temporal aggregation layers."""
        self.temporal_aggregation_layers = []

        for should_use_netvlad, clusters in zip(
            self.experts_use_netvlad, self.experts_netvlad_shape):
            
            self.temporal_aggregation_layers.append(TemporalAggregationLayer(
                self.expert_aggregated_size,
                should_use_netvlad,
                netvlad_clusters=clusters))

    def make_mlp(self, num_layers):
        """Makes and returns an mlp with num_layers layers."""
        sequential_layers = []
        
        for i in range(num_layers):
            sequential_layers.append(
                tf.keras.layers.Dense(
                    self.expert_aggregated_size,
                    activation=None))

            if i == num_layers - 1:
                break

            if self.use_batch_norm:
                sequential_layers.append(
                    tf.keras.layers.BatchNormalization(momentum=0.1))

            sequential_layers.append(
                self.make_activation_layer())

        return tf.keras.Sequential(sequential_layers)

    def make_gem_layers(self):
        """Create gated embedding module layers."""
        self.gems = []
        
        for _ in self.experts:
            self.gems.append(GatedEmbeddingUnitReasoning(
                self.expert_aggregated_size,
                self.encoded_expert_dimensionality))

    def call(self, inputs):
        """Forward pass on the video encoder."""
        assert len(inputs) == 2

        expert_embeddings, missing_experts = inputs

        assert len(expert_embeddings) == len(self.experts)

        aggregated_embeddings = self.temporal_aggreagation(expert_embeddings)
        output_embedding = self.collaborative_gating(
            aggregated_embeddings, missing_experts)

        return output_embedding

    def temporal_aggreagation(self, inputs):
        """Run temporal aggregation module."""
        aggregated_embeddings = []

        for aggregator, embedding in zip(
            self.temporal_aggregation_layers, inputs):
            aggregated_embeddings.append(aggregator(embedding))

        return aggregated_embeddings

    def collaborative_gating(self, aggregated_embeddings, missing_experts):
        """Run collaboartive gating module."""
        gated_embeddings = []
        experts_availability = 1 - tf.cast(missing_experts, tf.float32)

        for expert_index, embedding in enumerate(aggregated_embeddings):
            summed_attentions = 0
            experts_used = 0

            expert_available = experts_availability[:, expert_index]

            for other_expert_index, other_embedding in enumerate(
                aggregated_embeddings):
                attention_available = expert_available * experts_availability[
                    :, other_expert_index]
                
                if other_expert_index == expert_index:
                    continue

                attention = self.g_mlp(tf.concat(
                    [embedding, other_embedding],
                    axis=1,
                ))

                attention_available = tf.expand_dims(attention_available, -1)
                attention = attention * attention_available 
                experts_used = experts_used + attention_available
                summed_attentions = summed_attentions + attention_available

            attentions = tf.math.divide_no_nan(summed_attentions, experts_used) 
            attentions = self.h_mlp(attentions)

            #embedding = self.expert_projection([embedding, attentions])

            gated_embedding = self.gems[expert_index]([embedding, attentions])

            gated_embeddings.append(gated_embedding)

        return gated_embeddings
