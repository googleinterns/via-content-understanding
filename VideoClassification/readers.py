# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provides readers configured for different datasets."""

import tensorflow as tf
import utils
import os
from functools import partial

def resize_axis(tensor, axis, new_size, fill_value=0):
	"""Truncates or pads a tensor to new_size on on a given axis.

	Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
	size increases, the padding will be performed at the end, using fill_value.

	Args:
		tensor: The tensor to be resized.
		axis: An integer representing the dimension to be sliced.
		new_size: An integer or 0d tensor representing the new value for
			tensor.shape[axis].
		fill_value: Value to use to fill any new entries in the tensor. Will be cast
			to the type of tensor.

	Returns:
		The resized tensor.
	"""
	tensor = tf.convert_to_tensor(tensor)
	shape = tf.unstack(tf.shape(tensor))

	pad_shape = shape[:]
	pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

	shape[axis] = tf.minimum(shape[axis], new_size)
	shape = tf.stack(shape)

	resized = tf.concat([
			tf.slice(tensor, tf.zeros_like(shape), shape),
			tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
	], axis)

	# Update shape.
	new_shape = tensor.get_shape().as_list()  # A copy is being made.
	new_shape[axis] = new_size
	resized.set_shape(new_shape)
	return resized

class YT8MFrameFeatureDataset():
	"""Reads TFRecords of SequenceExamples.

	The TFRecords must contain SequenceExamples with the sparse in64 'labels'
	context feature and a fixed length byte-quantized feature vector, obtained
	from the features in 'feature_names'. The quantized features will be mapped
	back into a range between min_quantized_value and max_quantized_value.
	"""

	def __init__(  # pylint: disable=dangerous-default-value
			self,
			num_classes=3862,
			feature_sizes=[1024, 128],
			feature_names=["rgb", "audio"],
			max_frames=300):
		"""Construct a YT8MFrameFeatureDataset.

		Args:
			num_classes: a positive integer for the number of classes.
			feature_sizes: positive integer(s) for the feature dimensions as a list.
			feature_names: the feature name(s) in the tensorflow record as a list.
			max_frames: the maximum number of frames to process.
		"""

		assert len(feature_names) == len(feature_sizes), (
				"length of feature_names (={}) != length of feature_sizes (={})".format(
						len(feature_names), len(feature_sizes)))

		self.num_classes = num_classes
		self.feature_sizes = feature_sizes
		self.feature_names = feature_names
		self.max_frames = max_frames

	def get_video_matrix(self, features, feature_size, max_frames,
											 max_quantized_value, min_quantized_value):
		"""Decodes features from an input string and quantizes it.

		Args:
			features: raw feature values
			feature_size: length of each frame feature vector
			max_frames: number of frames (rows) in the output feature_matrix
			max_quantized_value: the maximum of the quantized value.
			min_quantized_value: the minimum of the quantized value.

		Returns:
			feature_matrix: matrix of all frame-features
			num_frames: number of frames in the sequence
		"""
		decoded_features = tf.reshape(
				tf.cast(tf.io.decode_raw(features, tf.uint8), tf.float32),
				[-1, feature_size])

		num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
		feature_matrix = utils.Dequantize(decoded_features, max_quantized_value,
																			min_quantized_value)
		feature_matrix = resize_axis(feature_matrix, 0, max_frames)
		return feature_matrix, num_frames

	def get_dataset(self, data_dir, batch_size, type="train", max_quantized_value=2, min_quantized_value=-2, num_workers=8):
		"""Returns TFRecordDataset after it has been parsed.

		Args:
			data_dir: directory of the TFRecords
		Returns:
			dataset: TFRecordDataset of the input training data
		"""
		files = tf.io.matching_files(os.path.join(data_dir, '%s*' % type))
		
		files_dataset = tf.data.Dataset.from_tensor_slices(files)
		files_dataset = files_dataset.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
		#files_dataset = files_dataset.repeat()
		dataset = files_dataset.interleave(tf.data.TFRecordDataset)
		dataset = dataset.shuffle(buffer_size=batch_size)

		parser = partial(self._parse_fn, max_quantized_value=max_quantized_value, min_quantized_value=min_quantized_value)
		dataset = dataset.map(parser)
		dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(batch_size)

		return dataset

	def _parse_fn(self, serialized_example, max_quantized_value=2, min_quantized_value=-2):
		"""Parse single Serialized Example from the TFRecords."""

		# Read/parse frame/segment-level labels.
		context_features = {
				"id": tf.io.FixedLenFeature([], tf.string),
				"labels": tf.io.VarLenFeature(tf.int64)
		}
		
		sequence_features = {
				feature_name: tf.io.FixedLenSequenceFeature([], dtype=tf.string)
				for feature_name in self.feature_names
		}
		contexts, features = tf.io.parse_single_sequence_example(
				serialized_example,
				context_features=context_features,
				sequence_features=sequence_features)

		# loads (potentially) different types of features and concatenates them
		num_features = len(self.feature_names)
		assert num_features > 0, "No feature selected: feature_names is empty!"

		assert len(self.feature_names) == len(self.feature_sizes), (
				"length of feature_names (={}) != length of feature_sizes (={})".format(
						len(self.feature_names), len(self.feature_sizes)))

		num_frames = -1  # the number of frames in the video
		feature_matrices = [None] * num_features  # an array of different features
		for feature_index in range(num_features):
			feature_matrix, num_frames_in_this_feature = self.get_video_matrix(
					features[self.feature_names[feature_index]],
					self.feature_sizes[feature_index], self.max_frames,
					max_quantized_value, min_quantized_value)
			if num_frames == -1:
				num_frames = num_frames_in_this_feature

			feature_matrices[feature_index] = feature_matrix

		# cap the number of frames at self.max_frames
		num_frames = tf.minimum(num_frames, self.max_frames)

		# concatenate different features
		video_matrix = tf.concat(feature_matrices, 1)

		
		# Process video-level labels.
		label_indices = contexts["labels"].values
		sparse_labels = tf.sparse.SparseTensor(
				tf.expand_dims(label_indices, axis=-1),
				tf.ones_like(contexts["labels"].values, dtype=tf.bool),
				(self.num_classes,))
		labels = tf.sparse.to_dense(sparse_labels, default_value=False, validate_indices=False)
		# convert to batch format.
		batch_video_ids = tf.expand_dims(contexts["id"], 0)
		batch_video_matrix = video_matrix
		batch_labels = labels
		batch_frames = tf.expand_dims(num_frames, 0)

		feature_dim = len(batch_video_matrix.get_shape()) - 1

		batch_video_matrix = tf.nn.l2_normalize(batch_video_matrix, feature_dim)

		#Add number of frames to data
		batch_frames = tf.repeat(tf.cast(tf.expand_dims(batch_frames, 1), tf.float32), repeats=[300], axis=0)
		batch_video_matrix = tf.stack([batch_video_matrix, batch_frames], axis=1)


		return (batch_video_matrix, batch_labels)
