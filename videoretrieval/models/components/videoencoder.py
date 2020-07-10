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

    This model takes in a list of features computed by pretrained expert models
    and produces a fixed length, sharded embeedding. This embedding corresponds
    to the embedding produced by a corresponding text encoder. This model should
    be trained in concert with a video encoder.

    The expert features are first aggregated down to a fixed length vector.
    Then, a model is used to compute an attention mask for each expert feature.
    The mask and embedding are fed into a Gated Embedding Unit for Video
    Reasoning.

    Attributes:
        num_experts: the number of pretrained experts used.
        expert_aggregated_size: the dimensionality experts features are
            projected to.
        encoded_expert_dimensionality: the dimensionality aggregated experts
            embeddings are mapped to.
        temporal_aggregation_layers: a list of temporal aggregation layers, one
            per expert.
        g_mlp: A standard feedforward deep neural network, with g_mlp_layers.
            This model takes in two embeddings and produces an attention mask.
        h_mlp: A standard feedforward deep neural network, with h_mlp_layers.
        gems: A list of gated embedding modules, one per embedding.
        activation_layer: the type of activation to be used.
        use_batch_norm: a boolean indicating if batch normalization is used.
    """

    def __init__(self, 
            num_experts,
            experts_use_netvlad,
            experts_netvlad_shape,
            expert_aggregated_size=768,
            encoded_expert_dimensionality=100,
            g_mlp_layers=5,
            h_mlp_layers=5,
            make_activation_layer=tf.keras.layers.ReLU,
            use_batch_norm=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros"
            ):
        """Initialize video encoder.

        Parameters:
            num_experts: the number of pretrained experts used, as an integer.
            expert_aggregated_size: the dimensionality experts are projected to.
            encoded_expert_dimensionality: the dimensionality experts embeddings
                are computed down to. Final output size is num of experts * 
                encoded_expert_dimensionality.
            g_mlp_layers: layers in the mlp labeled "g".
            h_mlp_layers: layers in the mlp labeled "h".
            use_batch_norm: if this model should use batch norm in the reasoning
                layers.
            kernel_initializer: the strategy used to initialize the weights in
                dense layer's kernel. Either a string naming the initializer or
                an instance of tf.keras.initializers.Initializer.
            bias_initial: the strategy used to initialize the weights in dense
                layers' biases. Either a string naming the initializer or
                an instance of tf.keras.initializers.Initializer.
        """

        super(VideoEncoder, self).__init__()

        self.num_experts = num_experts
        self.experts_use_netvlad = experts_use_netvlad
        self.experts_netvlad_shape = experts_netvlad_shape
        self.expert_aggregated_size = expert_aggregated_size
        self.encoded_expert_dimensionality = encoded_expert_dimensionality
        self.make_activation_layer = make_activation_layer
        self.use_batch_norm = use_batch_norm

        self.make_temporal_aggregation_layers(
            kernel_initializer, bias_initializer)

        self.g_mlp = self.make_mlp(
            g_mlp_layers, kernel_initializer, bias_initializer)
        self.h_mlp = self.make_mlp(
            h_mlp_layers, kernel_initializer, bias_initializer)

        self.make_gem_layers(kernel_initializer, bias_initializer)

    def make_temporal_aggregation_layers(
        self, kernel_initializer, bias_initializer):
        """Make temporal aggregation layers."""
        self.temporal_aggregation_layers = []

        for should_use_netvlad, clusters in zip(
            self.experts_use_netvlad, self.experts_netvlad_shape):
            
            self.temporal_aggregation_layers.append(TemporalAggregationLayer(
                self.expert_aggregated_size,
                should_use_netvlad,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                netvlad_clusters=clusters))

    def make_mlp(self, num_layers, kernel_initializer, bias_initializer):
        """Makes and returns an sequential feed forward neural network.

        This network is comprised of dense layers, batch normalization layers
        (if specified), and non-linearity functions. All dense layers except the
        final dense layer are followed by a batch normalization layer (if
        specified) and a non-linearity. The network ends with a single dense
        layer with a linear activation function.

        Parameters:
            num_layers: the number of dense layers in the sequential model.

        Returns: a sequential feed forward neural network.
        """

        sequential_layers = []
        
        for dense_layer_count in range(num_layers):
            sequential_layers.append(
                tf.keras.layers.Dense(
                    self.expert_aggregated_size,
                    activation=None,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer))

            if dense_layer_count == num_layers - 1:
                break

            if self.use_batch_norm:
                sequential_layers.append(
                    tf.keras.layers.BatchNormalization(momentum=0.1))

            sequential_layers.append(
                self.make_activation_layer())

        return tf.keras.Sequential(sequential_layers)

    def make_gem_layers(self, kernel_initializer, bias_initializer):
        """Create gated embedding reasoning units and adds them to self.gems."""
        self.gems = []
        
        for _ in range(self.num_experts):
            self.gems.append(GatedEmbeddingUnitReasoning(
                self.encoded_expert_dimensionality,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer))

    def call(self, inputs):
        """Forward pass on the video encoder.

        Parameters:
            inputs: inputs is a pair of two elements. First, a list of video
            experts. Second, a boolean tensor indicating missing
            video experts.

        Returns: an array of tensors, the ith tensor corresponding to the
        embedding shard from the ith video expert."""
        assert len(inputs) == 2

        expert_embeddings, missing_experts = inputs

        assert len(expert_embeddings) == self.num_experts

        aggregated_embeddings = self.temporal_aggregation(expert_embeddings)
        output_embedding = self.collaborative_gating(
            aggregated_embeddings, missing_experts)

        return output_embedding

    def temporal_aggregation(self, inputs):
        """Runs the temporal aggregation module.

        Parameters:
            inputs: a list of tensors from video experts.

        Returns: a list of temporally aggregated tensors, each dimension
        batch_size x self.expert_aggregated_size.
        """

        aggregated_embeddings = []

        for aggregator, embedding in zip(
            self.temporal_aggregation_layers, inputs):
            aggregated_embeddings.append(aggregator(embedding))

        return aggregated_embeddings

    def collaborative_gating(self, aggregated_embeddings, missing_experts):
        """Runs collaborative gating module.

        For each aggregated embedding, runs the feedfoward network in
        self.g_mlp to create an attention mask relative to each other embedding.
        These masks are then summed and scaled by the number of available 
        experts for a given example. Then, the aggregated embedding and scaled
        mask is fed into a gated embedding reasoning unit, which produces a
        normalized shard of the final embedding. 

        Parameters:
            aggregated_embeddings: a list of n aggregated video embeddings,
                where n is the number of video experts used. Each element of the
                list should be a tensor of shape batch_size x
                self.expert_aggregated_size.
            missing_experts: a tensor of shape batch_size x n, where n is the
                number of video experts. A "false" in this tensor indicates that
                the expert is not missing, while a true indicates the expert is
                missing.
        
        Returns: a list of tensors, each batch_size x
        self.encoded_expert_dimensionality. The ith tensor contains the parts
        of the embeddings from the ith expert."""

        gated_embeddings = []
        experts_availability = 1 - tf.cast(missing_experts, tf.float32)

        for expert_index, embedding in enumerate(aggregated_embeddings):
            summed_attentions = 0
            experts_used = 0

            expert_available = experts_availability[:, expert_index]

            for other_expert_index, other_embedding in enumerate(
                aggregated_embeddings):

                # if either the expert at expert_index or the expert at
                # other_expert_index is not available, the attention mask isn't
                # used

                attention_available = expert_available * experts_availability[
                    :, other_expert_index]
                
                attention = self.g_mlp(tf.concat(
                    [embedding, other_embedding],
                    axis=1,
                ))

                attention_available = tf.expand_dims(attention_available, -1)

                # This zeros out missing experts.
                attention = attention * attention_available 

                experts_used = experts_used + attention_available
                summed_attentions = summed_attentions + attention

            # This division scales the summed attentions by the number of experts
            # used, helping the model be more robust
            attentions = tf.math.divide_no_nan(summed_attentions, experts_used) 
            attentions = self.h_mlp(attentions)

            gated_embedding = self.gems[expert_index]([embedding, attentions])

            gated_embedding = gated_embedding * tf.expand_dims(
                expert_available, -1) 
            gated_embeddings.append(gated_embedding)

        return gated_embeddings
