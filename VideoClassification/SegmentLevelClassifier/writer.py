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

Writer file used to serialize and save data to TFRecords files.
"""
import numpy as np
import os
import pandas as pd
import tensorflow as tf

def add_candidate_content(context, candidates):
  """Add the tensor for the classes this particular video is a candidate for.
  
  Args:
    context: context of the video
    candidates: dictionary of candidates. Key is video id and value is list of candidate classes
  """
  video_id = tf.convert_to_tensor(context["id"])[0].numpy()
  if video_id in candidates.keys():
    context["candidate_labels"] = tf.convert_to_tensor(candidates[video_id])
  else:
    context["candidate_labels"] = tf.convert_to_tensor([])
  return context

def convert_labels(labels, class_csv="vocabulary.csv"):
  """Convert labels from range [0,3861] to range [0,1000]
  
  Args:
    labels: Tensor of labels to be converted
    class_csv: csv file containing conversion details
  """
  class_dataframe = pd.read_csv(class_csv, index_col=0)
  class_indices = class_dataframe.index.tolist()
  class_indices = np.array(class_indices)

  labels = labels.numpy()
  new_labels = []
  for label in labels:
    check = np.nonzero(class_indices == label)
    if np.any(check):
      new_label = check[0].tolist()[0]
      new_labels.append(new_label)
  return tf.convert_to_tensor(new_labels)

def convert_to_feature(item, type):
  """Convert item to FeatureList.

  Args:
    item: item to be converted
    type: string denoting the type of item. Can be "float", "byte", or "int"
  """
  if type == "float":
    item = tf.train.FloatList(value=item)
    item = tf.train.Feature(float_list=item)
  elif type == "byte":
    item = tf.train.BytesList(value=item)
    item = tf.train.Feature(bytes_list=item)
  elif type == "int":
    item = tf.train.Int64List(value=item)
    item = tf.train.Feature(int64_list=item)
  else:
    print("Invalid type entered for converting feature")
  return item

def serialize_features(features):
  """Serialize features.

  Args:
    features: features of the video
  """
  audio = features["audio"][0].numpy().tostring()
  rgb = features["rgb"][0].numpy().tostring()
  audio = convert_to_feature([audio], "byte")
  rgb = convert_to_feature([rgb], "byte")
  features = {"audio": tf.train.FeatureList(feature=[audio]), "rgb": tf.train.FeatureList(feature=[rgb])}
  features = tf.train.FeatureLists(feature_list=features)
  return features

def serialize_context(context):
  """Serialize context.

  Args:
    context: context of the video
  """
  video_id = tf.convert_to_tensor(context["id"])[0]
  labels = context["labels"].values
  segment_labels = context["segment_labels"].values
  segment_start_times = context["segment_start_times"].values
  segment_scores = context["segment_scores"].values
  candidate_labels = context["candidate_labels"]
  labels =  convert_labels(labels)
  segment_labels = convert_labels(segment_labels)

  context["id"] = convert_to_feature([video_id.numpy()], "byte")
  context["labels"] = convert_to_feature(labels.numpy(), "int")
  context["segment_labels"] = convert_to_feature(segment_labels.numpy(), "int")
  context["segment_start_times"] = convert_to_feature(segment_start_times.numpy(), "int")
  context["segment_scores"] = convert_to_feature(segment_scores.numpy(), "float")
  context["candidate_labels"] = convert_to_feature(candidate_labels.numpy(), "int")

  context = tf.train.Features(feature=context)
  return context

def serialize_video(context, features):
  """Serialize video from context and features.

  Args:
    context: context of the video
    features: features of the video
  """
  features = serialize_features(features)
  context = serialize_context(context)
  example = tf.train.SequenceExample(feature_lists=features, context=context)
  return example.SerializeToString()

def save_data(new_data_dir, input_dataset, candidates, file_type="validate", shard_size=17):
  """Save data as TFRecords Datasets in new_data_dir.

  Args:
    new_data_dir: string giving the directory to save TFRecords Files
    input_dataset: original dataset before candidate generation
    candidates: list of lists where each inner list contains the class indices that the corresponding input data is a candidate for. len(candidates) == len(input_dataset)
  """
  shard_counter = 0
  shard_number = 0
  shard = []
  for video in input_dataset:
    context = video[0]
    features = video[1]
    context = add_candidate_content(context, candidates)
    serialized_video = serialize_video(context, features)
    shard.append(serialized_video)
    shard_counter += 1
    if shard_counter == shard_size:
      print(f"Processing shard number {shard_number}")
      shard = tf.convert_to_tensor(shard)
      shard_dataset = tf.data.Dataset.from_tensor_slices(shard)
      file_name = file_type + str(shard_number)
      file_path = os.path.join(new_data_dir, '%s.tfrecord' % file_name)
      writer = tf.data.experimental.TFRecordWriter(file_path)
      writer.write(shard_dataset)
      shard_counter = 0
      shard_number += 1
      shard = []
  #Handles overflow
  if shard_counter != 0:
    print(f"Processing shard number {shard_number}")
    shard = tf.convert_to_tensor(shard)
    shard_dataset = tf.data.Dataset.from_tensor_slices(shard)
    file_name = file_type + str(shard_number)
    file_path = os.path.join(new_data_dir, '%s.tfrecord' % file_name)
    writer = tf.data.experimental.TFRecordWriter(file_path)
    writer.write(shard_dataset)
    shard_counter = 0
    shard_number += 1
    shard = []