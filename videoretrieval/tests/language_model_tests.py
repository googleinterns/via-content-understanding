"""Tests for the video encoder, text encoder, and associated layers.

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

import unittest
import tensorflow as tf
from tensorflow.python.framework import random_seed
import numpy as np
from abc import ABC as AbstractClass

from models.components import VideoEncoder, TextEncoder
from models.layers import GatedEmbeddingUnitReasoning, GatedEmbeddingModule,\
    TemporalAggregationLayer, NetVLAD

random_seed.set_seed(1)

BATCH_SIZE = 2
FEATURE_SIZE = 15
OUTPUT_DIM = 10

NUM_EXPERTS = 3
EXPERT_ONE_SHAPE = (BATCH_SIZE, 20)
EXPERT_TWO_SHAPE = (BATCH_SIZE, 5)
EXPERT_THREE_SHAPE = (BATCH_SIZE, 3, 4)
EXPERT_NETVLAD_CLUSTERS = 5
EXPERT_AGGREGATED_SIZE = OUTPUT_DIM
MOCK_TEXT_EMBEDDING_SHAPE = (BATCH_SIZE, 5, 20)

MLP_LAYERS = 2

class CollaborativeExpertsTestCase(unittest.TestCase, AbstractClass):
    """An class with helper methods for test of collaborative experts."""
    error_margin = 1e-6

    def assert_last_axis_has_norm(self, vector, norm=1):
        last_dimension_magnitude = tf.norm(vector, ord=2, axis=-1)
        expected_norm = norm * tf.ones_like(last_dimension_magnitude)

        error = abs(
            tf.reduce_max(last_dimension_magnitude - expected_norm, axis=-1))

        self.assertTrue(error <= self.error_margin)

    def assert_vector_has_shape(self, vector, shape):
        self.assertTrue(tuple(vector.shape) == tuple(shape))

class TestCollaborativeExpertsModels(CollaborativeExpertsTestCase):

    def test_video_encoder(self):
        """Tests initializing a video encoder and making a forward pass."""

        video_encoder = VideoEncoder(
            num_experts=NUM_EXPERTS,
            experts_use_netvlad=[False, False, True],
            experts_netvlad_shape=[None, None, EXPERT_NETVLAD_CLUSTERS],
            expert_aggregated_size=EXPERT_AGGREGATED_SIZE,
            encoded_expert_dimensionality=EXPERT_AGGREGATED_SIZE,
            g_mlp_layers=2)
        
        expert_one_data = tf.random.normal(EXPERT_ONE_SHAPE)
        expert_two_data = tf.random.normal(EXPERT_TWO_SHAPE)
        expert_three_data = tf.random.normal(EXPERT_THREE_SHAPE)

        missing_experts = tf.constant(
            [[False, False, False],
            [False, True, False]])

        outputs = video_encoder(
            (
                [expert_one_data, expert_two_data, expert_three_data],
                missing_experts))

        for embedding_shard in outputs:
            self.assert_vector_has_shape(
                embedding_shard, (BATCH_SIZE, EXPERT_AGGREGATED_SIZE))
            self.assert_last_axis_has_norm(embedding_shard, norm=1)


    def test_text_encoder(self):
        """Tests initializing a text encoder and making a forward pass."""
        text_encoder = TextEncoder(
            num_experts=NUM_EXPERTS,
            language_model_dimensionality=MOCK_TEXT_EMBEDDING_SHAPE[-1]
            encoded_expert_dimensionality=EXPERT_AGGREGATED_SIZE)

        mock_text_embeddings = tf.random.normal(MOCK_TEXT_EMBEDDING_SHAPE)
        embeddings, mixture_weights = mock_text_embeddings(mock_text_embeddings)

        self.assert_vector_has_shape(
            embeddings, (BATCH_SIZE, EXPERT_AGGREGATED_SIZE))
        self.assert_last_axis_has_norm(
            embeddings, norm=1)
        self.assert_last_axis_has_norm(
            mixture_weights, norm=1)


class TestCollaborativeExpertsLayers(CollaborativeExpertsTestCase):
    """Tests the layers used in the Collaborative Experts models."""


    def test_gated_embedding_unit_reasoning(self):
        """Tests a gated embedding unit for video reasoning."""
        gated_embedding_unit_reasoning = GatedEmbeddingUnitReasoning(
            FEATURE_SIZE,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros")

        mock_video_embedding = tf.random.normal((BATCH_SIZE, FEATURE_SIZE))
        mock_video_mask = tf.random.normal((BATCH_SIZE, FEATURE_SIZE))

        output = gated_embedding_unit_reasoning(
            [mock_video_embedding, mock_video_mask])

        self.assert_vector_has_shape(output, (BATCH_SIZE, FEATURE_SIZE))
        self.assert_last_axis_has_norm(output, norm=1)

    def test_gated_embedding_module(self):
        """Tests a gated embedding module layer."""
        gated_embedding_unit_reasoning = GatedEmbeddingUnitReasoning(
            FEATURE_SIZE,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros")

        mock_text_embedding = tf.random.normal((BATCH_SIZE, FEATURE_SIZE))

        output = gated_embedding_unit_reasoning(mock_text_embedding)

        self.assert_vector_has_shape(output, (BATCH_SIZE, FEATURE_SIZE))
        self.assert_last_axis_has_norm(output, norm=1)


    def test_temporal_aggregation_layer(self):
        """Tests a temporal aggregation layer."""

        temporal_aggregation_layer_no_netvlad = TemporalAggregationLayer(
            output_dim=OUTPUT_DIM,
            use_netvlad=False,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros")

        random_inputs = tf.random.normal((BATCH_SIZE, FEATURE_SIZE))
        output = temporal_aggregation_layer_no_netvlad(random_inputs)

        self.assert_vector_has_shape(output, (BATCH_SIZE, OUTPUT_DIM))
        self.assert_last_axis_has_norm(output, norm=1)

        temporal_aggregation_layer_with_netvlad = TemporalAggregationLayer(
            output_dim=OUTPUT_DIM,
            use_netvlad=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros")

        random_inputs = tf.random.normal((BATCH_SIZE, 5, 200))
        output = temporal_aggregation_layer_with_netvlad(random_inputs)

        self.assert_vector_has_shape(output, (BATCH_SIZE, OUTPUT_DIM))
        self.assert_last_axis_has_norm(output, norm=1)
