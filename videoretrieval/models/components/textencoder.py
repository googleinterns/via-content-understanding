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

from tensorflow_addons.layers.netvlad import NetVLAD
from models.layers import GatedEmbeddingModule

class TextEncoder(tf.keras.Model):
	"""Implementation of TextEncoder."""
	def __init__(self, 
            num_of_experts,
            num_netvlad_clusters=25,
            language_model_dimensionality=768,
            encoded_expert_dimensionality=100):
		"""Initalize TextEncoder.

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
		self.netvlad = NetVLAD(num_netvlad_clusters)
		self.encoded_expert_dimensionality = encoded_expert_dimensionality

		self.make_gems()
		self.make_dense_layers()

	def make_gems(self):
		"""Initalize gated embedding modules."""
		self.gems = []

		for _ in range(self.num_of_experts):
			self.gems.append(GatedEmbeddingModule(
				self.language_model_dimensionality * self.num_netvlad_clusters,
				self.encoded_expert_dimensionality))


	def make_dense_layers(self):
		"""Make dense + softmax layers."""
		self.dense_layers = []

		for _ in range(self.num_of_experts):
			self.dense_layers.append(tf.keras.layers.Dense(
				self.encoded_expert_dimensionality,
				activation="softmax"
				))

	def call(self, input_):
		"""Forward pass."""

		aggregated_embeddings = self.netvlad(input_)

		expert_embeddings = []

		for i in range(self.num_of_experts):
			gated_embeddings = self.gems[i](aggregated_embeddings)
			expert_embeddings.append(self.dense_layers[i](gated_embeddings))

		return tf.concat(expert_embeddings, axis=1)
