"""
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
import model_utils as utils
import math

class NetVLAD(tf.keras.layers.Layer):
	"""Applies NetVLAD to the input.
	Arguments:
		num_clusters: The number of clusters to use.
	Input shape:
		3D tensor with shape: `(batch_size, time, feature_dim)`.
	Output shape:
		2D tensor with shape: `(batch_size, feature_dim * num_clusters)`.
	"""
	def __init__(self, num_clusters, input_shape, **kwargs):
		super().__init__(**kwargs)
		if num_clusters <= 0:
			raise ValueError("`num_clusters` must be greater than 1: %i" % num_clusters)
		self.num_clusters = num_clusters
		feature_dim = input_shape[-1]
		if not isinstance(feature_dim, int):
			feature_dim = feature_dim.value
		self.fc = tf.keras.layers.Dense(
			units=self.num_clusters,
			activation=tf.nn.softmax,
			kernel_regularizer=tf.keras.regularizers.l2(1e-5),
			name="vlad_fc" + str(num_clusters)
		)
		self.cluster_centers = self.add_weight(
			shape=(1, feature_dim, self.num_clusters),
			initializer=tf.keras.initializers.TruncatedNormal(
				stddev=1.0 / math.sqrt(feature_dim)
			),
			trainable=True,
			name="cluster_centers" + str(num_clusters)
		)
		self.feature_dim = feature_dim
		self.max_frames = input_shape[-2]

	def call(self, frames):
		"""Apply the NetVLAD module to the given frames.
		Args:
			frames: A tensor with shape [batch_size, max_frames, feature_dim].
		Returns:
			A tensor with shape [batch_size, feature_dim * num_clusters].
		Raises:
			ValueError: If the `feature_dim` of input is not defined.
		"""
		feature_dim = self.feature_dim
		max_frames = self.max_frames
		
		frames = tf.reshape(frames, (-1, feature_dim))
		activation = self.fc(frames)
		activation = tf.reshape(activation, (-1, max_frames, self.num_clusters))
		
		a_sum = tf.math.reduce_sum(activation, axis=-2, keepdims=True)
		a = a_sum * self.cluster_centers

		frames = tf.reshape(frames, (-1, max_frames, feature_dim))
		b = tf.transpose(
			tf.matmul(tf.transpose(activation, perm=(0, 2, 1)), frames), perm=(0, 2, 1)
		)
		
		vlad = b - a
		vlad = tf.nn.l2_normalize(vlad, 1)
		vlad = tf.reshape(vlad, (-1, feature_dim * self.num_clusters))
		vlad = tf.nn.l2_normalize(vlad, 1)

		return vlad

	def compute_output_shape(self, input_shape):
		input_shape = tf.TensorShape(input_shape).as_list()
		return tf.TensorShape([input_shape[0], input_shape[-1] * self.num_clusters])

	def get_config(self):
		config = {"num_clusters": self.num_clusters}
		base_config = super().get_config()
		return dict(list(base_config.items()) + list(config.items()))

class ContextGating(tf.keras.layers.Layer):
	"""Implements the Context Gating Layer from https://arxiv.org/abs/1706.06905

	Input shape:
	2D tensor with shape: `(batch_size, feature_dim)`.
	Output shape:
	2D tensor with shape: `(batch_size, feature_dim)`.
	"""

	def __init__(self, input_shape, **kwargs):
		super(ContextGating, self).__init__(**kwargs)
		feature_dim = input_shape[-1]
		if not isinstance(feature_dim, int):
			feature_dim = feature_dim.value
		self.fc = tf.keras.layers.Dense(
			units=feature_dim,
			activation=tf.nn.sigmoid,
			kernel_regularizer=tf.keras.regularizers.l2(1e-5),
		)

	def call(self, model_input):
		"""Apply the ContextGating module to the given input.

		Args:
			model_input: A tensor with shape [batch_size, feature_dim].
		Returns:
			A tensor with shape [batch_size, feature_dim].
		Raises:
			ValueError: If the `feature_dim` of model_input is not defined.
		"""
		model_input.shape.assert_has_rank(2)
		feature_dim = model_input.shape.as_list()[-1]
		if feature_dim is None:
			raise ValueError("Last dimension must be defined.")
		
		context_gate = self.fc(model_input)

		output = tf.math.multiply(context_gate, model_input)

		return output

	def compute_output_shape(self, input_shape):
		return input_shape

	def get_config(self):
		base_config = super().get_config()
		return dict(list(base_config.items()))

class MOELogistic(tf.keras.layers.Layer):
	"""Implements a Mixture of Logistic Experts classifier.

	Input shape:
		2D tensor with shape: `(batch_size, feature_dim)`.
	Output shape:
		2D tensor with shape: `(batch_size, num_classes)`.
	"""

	def __init__(self, input_shape, num_classes, num_mixtures, **kwargs):
		super(MOELogistic, self).__init__(**kwargs)
		self.num_classes = num_classes
		self.num_mixtures = num_mixtures

		self.gate_fc = tf.keras.layers.Dense(
			units=num_classes*(num_mixtures+1),
			kernel_regularizer=tf.keras.regularizers.l2(1e-5),
		)

		self.expert_fc = tf.keras.layers.Dense(
			units=num_classes*num_mixtures,
			kernel_regularizer=tf.keras.regularizers.l2(1e-5),
		)

	def call(self, input):
		"""Apply the MoE algorithm to the given input.

		Args:
			input: A tensor with shape [batch_size, feature_dim].
		Returns:
			A tensor with shape [batch_size, num_classes].
		Raises:
			ValueError: If the `feature_dim` of input is not defined.
		"""
		gate_activations = self.gate_fc(input)
		expert_activations = self.expert_fc(input)

		#Calculate the distribution across mixtures
		gate_dist = tf.nn.softmax(tf.reshape(gate_activations, [-1, self.num_mixtures+1]))
		expert_dist = tf.nn.sigmoid(tf.reshape(expert_activations, [-1, self.num_mixtures]))

		probs = tf.reduce_sum(tf.math.multiply(gate_dist[:,:self.num_mixtures], expert_dist),1)
		probs = tf.reshape(probs, [-1, self.num_classes])

		return probs

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.num_classes)

	def get_config(self):
		base_config = super().get_config()
		config = base_config.update({'number of classes': self.num_classes, 'number of mixtures': self.num_mixtures})
		return config

class VideoClassifier(tf.keras.Model):
	"""The Video Classifier model, implemented according to the winning model from the Youtube-8M Challenge.
	The model can be found here: https://arxiv.org/pdf/1706.06905.pdf
	
	Arguments:
		num_clusters: the number of clusters to be used for NetVLAD. The audio clusters will be num_clusters/2.
		video_input_shape: shape of the input video features. Shape of [batch_size, num_samples, video_feature_dim].
		audio_input_shape: shape fo the input audio features. Shape of [batch_size, num_samples, audio_feature_dim].
	
	Raises:
		ValueError: If num_clusters is not divisible by 2.
		ValueError: If the batch sizes of the audio_input_shape and video_input_shape do not match.
		ValueError: If the number of samples of the audio_input_shape and video_input_shape do not match.
	"""
	def __init__(self, num_clusters, video_input_shape, audio_input_shape, num_classes, num_mixtures, fc_units, iterations, random_frames, **kwargs):
		super(VideoClassifier, self).__init__(**kwargs)
		if num_clusters % 2 != 0:
			raise ValueError("num_clusters must be divisible by 2.")
		batch_size = video_input_shape[0]
		if audio_input_shape[0] != batch_size:
			raise ValueError("audio_input_shape[0] must equal video_input_shape[0]. Batch sizes must equal.")
		if audio_input_shape[1] != video_input_shape[1]:
			raise ValueError("audio_input_shape[1] must equal video_input_shape[1]. Number of samples must equal.")

		self.num_frames = video_input_shape[1]
		self.num_classes = num_classes
		self.num_mixtures = num_mixtures
		self.iterations = iterations
		self.random_frames = random_frames

		self.video_feature_dim = video_input_shape[2]

		self.video_vlad = NetVLAD(num_clusters, input_shape=video_input_shape, name="video_vlad")
		self.audio_vlad = NetVLAD(num_clusters//2, input_shape=audio_input_shape, name="audio_vlad")

		#Relu6 is used as it is employed in the paper.
		self.fc = tf.keras.layers.Dense(
			units=fc_units,
			activation=tf.nn.relu6,
			kernel_regularizer=tf.keras.regularizers.l2(1e-5),
			name="main_fc"
		)

		self.first_cg = ContextGating(input_shape=(batch_size, fc_units), name="first_cg")

		self.moe = MOELogistic(input_shape=self.first_cg.compute_output_shape((batch_size, fc_units)), num_classes=self.num_classes, num_mixtures=self.num_mixtures, name="moe")

		#self.second_cg = ContextGating(input_shape=self.moe.compute_output_shape((batch_size, fc_units)), name="second_cg")

	def call(self, inputs):
		"""Perform one forward pass of the model.

		Args:
			model_input: input features of shape [batch_size, max_frames, video_feature_dim + audio_feature_dim + 1].
		Returns:
			A tensor with shape [batch_size, num_classes].
		"""
		print(inputs)
		model_input = inputs[0]
		num_frames = inputs[1]
		print(tf.shape(num_frames))
		num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
		if self.random_frames:
			model_input = utils.SampleRandomFrames(model_input, num_frames, self.iterations)
		else:
			model_input = utils.SampleRandomSequence(model_input, num_frames, self.iterations)

		video_input = model_input[:,:,:self.video_feature_dim]
		audio_input = model_input[:,:,self.video_feature_dim:]

		video_vlad_out = self.video_vlad(video_input)
		audio_vlad_out = self.audio_vlad(audio_input)

		vlad_out = tf.concat([video_vlad_out, audio_vlad_out], axis=1)

		fc_out = self.fc(vlad_out)
		cg_out = self.first_cg(fc_out)
		moe_out = self.moe(cg_out)
		#final_out = self.second_cg(moe_out)
		final_out = moe_out
		return final_out