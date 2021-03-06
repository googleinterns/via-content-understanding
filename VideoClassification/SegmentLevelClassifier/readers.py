"""Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

File containing dataset readers.
"""
from functools import partial
import os
import reader_utils as utils
import tensorflow as tf

class BaseDataset():
  """Base class for which all Dataset classes within readers.py inherit from.
  """

  def __init__(self, num_classes, feature_sizes, feature_names):
    """Construct a YT8MFrameFeatureDataset.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list. Must be same size as feature_names
      feature_names: the feature name(s) in the tensorflow record as a list. Must be same size as feature_sizes
    """

    assert len(feature_names) == len(feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(feature_names), len(feature_sizes)))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names

  def get_dataset(self, data_dir, batch_size=1, type="train"):
    """Returns TFRecordDataset after it has been parsed.

    Args:
      data_dir: directory of the TFRecords
    Returns:
      dataset: TFRecordDataset of the input training data
    """
    files = tf.io.matching_files(os.path.join(data_dir, '%s*.tfrecord' % type))
    
    files_dataset = tf.data.Dataset.from_tensor_slices(files)
    files_dataset = files_dataset.batch(tf.cast(tf.shape(files)[0], tf.int64))

    dataset = files_dataset.interleave(lambda files: tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.experimental.AUTOTUNE))

    parser = partial(self._parse_fn)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

class VideoDataset(BaseDataset):
  """Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized feature vector, obtained
  from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  """

  def __init__(
      self,
      num_classes=3862,
      feature_sizes=[1024, 128],
      feature_names=["rgb", "audio"],
      max_frames=300):
    """Construct a YT8MFrameFeatureDataset.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list. Must be same size as feature_names
      feature_names: the feature name(s) in the tensorflow record as a list. Must be same size as feature_sizes
      max_frames: the maximum number of frames to process.
    """

    super().__init__(num_classes, feature_sizes, feature_names)
    self.max_frames = max_frames

  def get_video_matrix(self, features, feature_size, max_frames, max_quantized_value, min_quantized_value):
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
    feature_matrix = utils.dequantize(decoded_features, max_quantized_value,
                                      min_quantized_value)

    feature_matrix = utils.resize_axis(feature_matrix, 0, max_frames)
    return feature_matrix, num_frames

  def _parse_fn(self, serialized_example, max_quantized_value=2, min_quantized_value=-2):
    """Parse single Serialized Example from the TFRecords."""
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

    num_features = len(self.feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"

    assert len(self.feature_names) == len(self.feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(self.feature_names), len(self.feature_sizes)))

    num_frames = -1
    feature_matrices = [None] * num_features
    for feature_index in range(num_features):
      feature_matrix, num_frames_in_this_feature = self.get_video_matrix(
          features[self.feature_names[feature_index]],
          self.feature_sizes[feature_index], self.max_frames,
          max_quantized_value, min_quantized_value)
      if num_frames == -1:
        num_frames = num_frames_in_this_feature

      feature_matrices[feature_index] = feature_matrix

    num_frames = tf.minimum(num_frames, self.max_frames)
    video_matrix = tf.concat(feature_matrices, 1)
    num_frames = tf.expand_dims(num_frames, 0)
    label_indices = contexts["labels"].values
    sparse_labels = tf.sparse.SparseTensor(
        tf.expand_dims(label_indices, axis=-1),
        tf.ones_like(contexts["labels"].values, dtype=tf.bool),
        (self.num_classes,))
    labels = tf.sparse.to_dense(sparse_labels, default_value=False, validate_indices=False)
    batch_video_ids = tf.expand_dims(contexts["id"], 0)
    
    batch_video_matrix = video_matrix
    batch_labels = labels

    feature_dim = len(batch_video_matrix.get_shape()) - 1
    batch_video_matrix = tf.nn.l2_normalize(batch_video_matrix, feature_dim)
    return (contexts["id"], batch_video_matrix, batch_labels)

class BasicDataset(BaseDataset):
  """Reads TFRecords of SequenceExamples for Segment level data. Used in candidate_generation.py

  The TFRecords must contain SequenceExamples with the sparse int64 'labels', string 'id', int64 'segment_labels',
  int64 segment_start_times', and float32 'segment_scores' context features and a fixed length byte-quantized
  feature vector, obtained from the features in 'feature_names'.
  """

  def __init__(
      self,
      num_classes=1000,
      feature_sizes=[1024, 128],
      feature_names=["rgb", "audio"]
      ):
    """Construct a BasicDataset.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list. Must be same size as feature_names
      feature_names: the feature name(s) in the tensorflow record as a list. Must be same size as feature_sizes
    """

    super().__init__(num_classes, feature_sizes, feature_names)

  def _parse_fn(self, serialized_example):
    """Parse single Serialized Example from the TFRecords."""
    # Read/parse frame/segment-level labels.
    context_features = {
      "id": tf.io.FixedLenFeature([], tf.string),
      "labels": tf.io.VarLenFeature(tf.int64),
      "segment_labels": tf.io.VarLenFeature(tf.int64),
      "segment_start_times": tf.io.VarLenFeature(tf.int64),
      "segment_scores": tf.io.VarLenFeature(tf.float32)
    }
    sequence_features = {
        feature_name: tf.io.FixedLenSequenceFeature([], dtype=tf.string)
        for feature_name in self.feature_names
    }
    context, features = tf.io.parse_single_sequence_example(serialized_example, context_features=context_features, sequence_features=sequence_features)

    features["rgb"] = tf.io.decode_raw(features["rgb"], tf.uint8)
    features["audio"] = tf.io.decode_raw(features["audio"], tf.uint8)
    return (context, features)

class SplitDataset(BaseDataset):
  """Reads TFRecords of SequenceExamples for Segment level data. Used for the input pipeline where data is split.

  The TFRecords must contain SequenceExamples with the sparse int64 'labels', string 'id', int64 'segment_labels',
  int64 segment_start_times', and float32 'segment_scores' context features and a fixed length byte-quantized
  feature vector, obtained from the features in 'feature_names'.
  """

  def __init__(
      self,
      pipeline_type="train",
      num_classes=1000,
      feature_sizes=[1024, 128],
      feature_names=["rgb", "audio"]
      ):
    """Construct a SplitDataset.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list. Must be same size as feature_names
      feature_names: the feature name(s) in the tensorflow record as a list. Must be same size as feature_sizes
    """

    super().__init__(num_classes, feature_sizes, feature_names)
    self.pipeline_type = pipeline_type

  def _parse_fn(self, serialized_example):
    """Parse single Serialized Example from the TFRecords."""
    # Read/parse frame/segment-level labels.
    context_features = {
      "id": tf.io.FixedLenFeature([], tf.string),
      "labels": tf.io.VarLenFeature(tf.int64),
      "segment_labels": tf.io.VarLenFeature(tf.int64),
      "segment_start_times": tf.io.VarLenFeature(tf.int64),
      "segment_scores": tf.io.VarLenFeature(tf.float32)
    }
    if self.pipeline_type == "test":
      context_features["candidate_labels"] = tf.io.VarLenFeature(tf.int64)

    sequence_features = {
        feature_name: tf.io.FixedLenSequenceFeature([], dtype=tf.string)
        for feature_name in self.feature_names
    }
    context, features = tf.io.parse_single_sequence_example(serialized_example, context_features=context_features, sequence_features=sequence_features)

    num_features = len(self.feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"

    assert len(self.feature_names) == len(self.feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(self.feature_names), len(self.feature_sizes)))

    feature_matrices = [None] * num_features
    for feature_index in range(num_features):
      feature_matrix = tf.reshape(tf.io.decode_raw(features[self.feature_names[feature_index]], tf.uint8), 
                                    [-1, self.feature_sizes[feature_index]])
      feature_matrices[feature_index] = feature_matrix

    features["rgb"] = feature_matrices[0]
    features["audio"] = feature_matrices[1]
    return (context, features)

class SegmentDataset(BaseDataset):
  """Reads TFRecords of SequenceExamples for Segment level data. Used for the input pipeline where class specific features are generated.

  The TFRecords must contain SequenceExamples with the string 'id', int64 'segment_label',
  int64 segment_start_time', and float32 'segment_score' context features and a fixed length byte-quantized
  feature vector, obtained from the features in 'feature_names'.
  """

  def __init__(
      self,
      num_classes=1000,
      feature_sizes=[1024, 128],
      feature_names=["rgb", "audio"],
      class_num=-1,
      pipeline_type="train"):
    """Construct a SegmentDataset.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list. Must be same size as feature_names
      feature_names: the feature name(s) in the tensorflow record as a list. Must be same size as feature_sizes
      class_num: determines which class file to read. To read all files, do not modify class_num.
    """

    super().__init__(num_classes, feature_sizes, feature_names)
    self.class_num = class_num
    self.pipeline_type = pipeline_type

  def get_dataset(self, data_dir, batch_size, type="train"):
    """Returns TFRecordDataset after it has been parsed.

    Args:
      data_dir: directory of the TFRecords
    Returns:
      dataset: TFRecordDataset of the input training data
    """
    if self.class_num == -1:
      files = tf.io.matching_files(os.path.join(data_dir, '%s*.tfrecord' % type))
    else:
      files = tf.io.matching_files(os.path.join(data_dir, '%s.tfrecord' % (type+str(self.class_num))))
    
    files_dataset = tf.data.Dataset.from_tensor_slices(files)
    files_dataset = files_dataset.batch(tf.cast(tf.shape(files)[0], tf.int64))

    dataset = files_dataset.interleave(lambda files: tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.experimental.AUTOTUNE))

    parser = partial(self._parse_fn)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(1).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

  def _parse_fn(self, serialized_example):
    """Parse single Serialized Example from the TFRecords."""
    # Read/parse frame/segment-level labels.
    context_features = {
      "id": tf.io.FixedLenFeature([], tf.string),
      "segment_label": tf.io.FixedLenFeature([], tf.int64),
      "segment_start_time": tf.io.FixedLenFeature([], tf.int64),
      "segment_score": tf.io.FixedLenFeature([], tf.float32)  
    }
    if self.pipeline_type == "test":
      context_features["candidate_label"] = tf.io.FixedLenFeature([], tf.int64)
      context_features["segment_id"] = tf.io.FixedLenFeature([], tf.int64)
    sequence_features = {
        feature_name: tf.io.FixedLenSequenceFeature([], dtype=tf.string)
        for feature_name in self.feature_names
    }
    context, features = tf.io.parse_single_sequence_example(serialized_example, context_features=context_features, sequence_features=sequence_features)

    num_features = len(self.feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"

    assert len(self.feature_names) == len(self.feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(self.feature_names), len(self.feature_sizes)))

    feature_matrices = [None] * num_features
    for feature_index in range(num_features):
      feature_matrix = tf.reshape(tf.io.decode_raw(features[self.feature_names[feature_index]], tf.uint8), 
                                    [-1, self.feature_sizes[feature_index]])
      feature_matrices[feature_index] = feature_matrix

    features["rgb"] = feature_matrices[0]
    features["audio"] = feature_matrices[1]
    return (context, features)

class CombineSegmentDataset(BaseDataset):
  """Reads TFRecords of SequenceExamples for Segment level data. Used for the input pipeline where class specific features are generated.

  The TFRecords must contain SequenceExamples with the string 'id', int64 'segment_label',
  int64 segment_start_time', and float32 'segment_score' context features and a fixed length byte-quantized
  feature vector, obtained from the features in 'feature_names'.
  """

  def __init__(
      self,
      num_classes=1000,
      feature_sizes=[1024, 128],
      feature_names=["rgb", "audio"],
      class_num=-1):
    """Construct a CombineSegmentDataset.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list. Must be same size as feature_names
      feature_names: the feature name(s) in the tensorflow record as a list. Must be same size as feature_sizes
      class_num: determines which class file to read. To read all files, do not modify class_num.
    """

    super().__init__(num_classes, feature_sizes, feature_names)
    self.class_num = class_num

  def get_dataset(self, data_dir, batch_size, type="train"):
    """Returns TFRecordDataset after it has been parsed.

    Args:
      data_dir: directory of the TFRecords
    Returns:
      dataset: TFRecordDataset of the input training data
    """
    if self.class_num == -1:
      files = tf.io.matching_files(os.path.join(data_dir, '%s*.tfrecord' % type))
    else:
      files = tf.io.matching_files(os.path.join(data_dir, '%s.tfrecord' % (type+str(self.class_num))))
    
    files_dataset = tf.data.Dataset.from_tensor_slices(files)
    files_dataset = files_dataset.batch(tf.cast(tf.shape(files)[0], tf.int64))

    dataset = files_dataset.interleave(lambda files: tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.experimental.AUTOTUNE))

    parser = partial(self._parse_fn)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(1).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

  def _parse_fn(self, serialized_example):
    """Parse single Serialized Example from the TFRecords."""
    # Read/parse frame/segment-level labels.
    context_features = {
      "id": tf.io.FixedLenFeature([], tf.string),
      "segment_label": tf.io.FixedLenFeature([], tf.int64),
      "segment_start_time": tf.io.FixedLenFeature([], tf.int64),
      "segment_score": tf.io.FixedLenFeature([], tf.float32)  
    }
    context_features["candidate_label"] = tf.io.FixedLenFeature([], tf.int64)
    context_features["segment_id"] = tf.io.FixedLenFeature([], tf.int64)
    sequence_features = {
        feature_name: tf.io.FixedLenSequenceFeature([], dtype=tf.string)
        for feature_name in self.feature_names
    }
    sequence_features["class_features"] = tf.io.FixedLenSequenceFeature([2], dtype=tf.float32)
    context, features = tf.io.parse_single_sequence_example(serialized_example, context_features=context_features, sequence_features=sequence_features)

    num_features = len(self.feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"

    assert len(self.feature_names) == len(self.feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(self.feature_names), len(self.feature_sizes)))

    feature_matrices = [None] * num_features
    for feature_index in range(num_features):
      feature_matrix = tf.reshape(tf.io.decode_raw(features[self.feature_names[feature_index]], tf.uint8), [-1, self.feature_sizes[feature_index]])
      feature_matrices[feature_index] = feature_matrix

    features["rgb"] = feature_matrices[0]
    features["audio"] = feature_matrices[1]
    return (context, features)

class InputDataset(BaseDataset):
  """Reads TFRecords of SequenceExamples for Segment level data. Used for input to the model

  The TFRecords must contain SequenceExamples with the string 'id', int64 'segment_label',
  int64 segment_start_time', and float32 'segment_score' context features and a fixed length byte-quantized
  feature vector, obtained from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  """

  def __init__(
      self,
      num_classes=1000,
      feature_sizes=[1024, 128, 2],
      feature_names=["rgb", "audio", "class_features"]):
    """Construct a InputDataset.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list. Must be same size as feature_names
      feature_names: the feature name(s) in the tensorflow record as a list. Must be same size as feature_sizes
    """

    super().__init__(num_classes, feature_sizes, feature_names)

  def get_video_matrix(self, features, feature_size, max_quantized_value, min_quantized_value):
    """Decodes features from an input string and quantizes it.

    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      feature_matrix: matrix of all frame-features
    """
    decoded_features = tf.reshape(
        tf.cast(tf.io.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    feature_matrix = utils.dequantize(decoded_features, max_quantized_value,
                                      min_quantized_value)
    return feature_matrix

  def _parse_fn(self, serialized_example, max_quantized_value=2, min_quantized_value=-2):
    """Parse single Serialized Example from the TFRecords."""
    context_features = {
      "id": tf.io.FixedLenFeature([], tf.string),
      "segment_label": tf.io.FixedLenFeature([], tf.int64),
      "segment_start_time": tf.io.FixedLenFeature([], tf.int64),
      "segment_score": tf.io.FixedLenFeature([], tf.float32)  
    }
    sequence_features = {
        feature_name: tf.io.FixedLenSequenceFeature([], dtype=tf.string)
        for feature_name in self.feature_names[:2]
    }
    sequence_features[self.feature_names[-1]] = tf.io.FixedLenSequenceFeature([2], dtype=tf.float32)
    context, features = tf.io.parse_single_sequence_example(serialized_example, context_features=context_features, sequence_features=sequence_features)
    num_features = len(self.feature_names)

    assert num_features > 0, "No feature selected: feature_names is empty!"
    assert len(self.feature_names) == len(self.feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(self.feature_names), len(self.feature_sizes)))
    
    feature_matrices = [None] * (num_features-1)
    for feature_index in range(num_features-1):
      feature_matrix = self.get_video_matrix(
        features[self.feature_names[feature_index]], self.feature_sizes[feature_index],
        max_quantized_value, min_quantized_value
        )
      feature_matrices[feature_index] = feature_matrix
    video_matrix = tf.concat(feature_matrices, 1)
    class_features_list = features[self.feature_names[2]]
    class_features_list = tf.reshape(class_features_list, [2,])
    segment_label = tf.reshape(tf.cast(context["segment_label"], tf.float32), [1,])
    class_features_list = tf.concat([segment_label, class_features_list],0)
    label = context["segment_score"]
    feature_dim = len(video_matrix.get_shape()) - 1
    video_matrix = tf.nn.l2_normalize(video_matrix, feature_dim)
    return ((video_matrix, class_features_list), label)

class EvaluationDataset(BaseDataset):
  """Reads TFRecords of SequenceExamples for Segment level data. Used for input to the model

  The TFRecords must contain SequenceExamples with the string 'id', int64 'segment_label',
  int64 segment_start_time', and float32 'segment_score' context features and a fixed length byte-quantized
  feature vector, obtained from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  """

  def __init__(
      self,
      num_classes=1000,
      feature_sizes=[1024, 128],
      feature_names=["rgb", "audio"]):
    """Construct a EvaluationDataset.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list. Must be same size as feature_names
      feature_names: the feature name(s) in the tensorflow record as a list. Must be same size as feature_sizes
    """
    super().__init__(num_classes, feature_sizes, feature_names)

  def get_video_matrix(self, features, feature_size, max_quantized_value, min_quantized_value):
    """Decodes features from an input string and quantizes it.

    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      feature_matrix: matrix of all frame-features
    """
    decoded_features = tf.reshape(
        tf.cast(tf.io.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    feature_matrix = utils.dequantize(decoded_features, max_quantized_value,
                                      min_quantized_value)
    return feature_matrix

  def _parse_fn(self, serialized_example, max_quantized_value=2, min_quantized_value=-2):
    """Parse single Serialized Example from the TFRecords."""
    context_features = {
      "id": tf.io.FixedLenFeature([], tf.string),
      "segment_label": tf.io.FixedLenFeature([], tf.int64)
    }
    sequence_features = {
        feature_name: tf.io.FixedLenSequenceFeature([], dtype=tf.string)
        for feature_name in self.feature_names
    }
    sequence_features["class_features"] = tf.io.VarLenFeature(dtype=tf.float32)
    context, features = tf.io.parse_single_sequence_example(serialized_example, context_features=context_features, sequence_features=sequence_features)
    num_features = len(self.feature_names)

    assert num_features > 0, "No feature selected: feature_names is empty!"
    assert len(self.feature_names) == len(self.feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(self.feature_names), len(self.feature_sizes)))
    
    feature_matrices = [None] * num_features
    for feature_index in range(num_features):
      feature_matrix = self.get_video_matrix(
        features[self.feature_names[feature_index]], self.feature_sizes[feature_index],
        max_quantized_value, min_quantized_value
        )
      feature_matrices[feature_index] = feature_matrix
    video_matrix = tf.concat(feature_matrices, 1)
    feature_dim = len(video_matrix.get_shape()) - 1
    video_matrix = tf.nn.l2_normalize(video_matrix, feature_dim)
    label = tf.one_hot(context["segment_label"], self.num_classes)
    class_features_list = features["class_features"]
    return ((video_matrix, class_features_list), label)