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
    context["candidate_labels"] = np.array(candidates[video_id])
  else:
    context["candidate_labels"] = np.array([])
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
    check = np.nonzero(class_indices == label)[0]
    if len(check) > 0:
      new_label = check.tolist()[0]
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

def serialize_class_features(features):
  """Serialize features.

  Args:
    features: features of the video
  """
  audio = features["audio"].tostring()
  rgb = features["rgb"].tostring()
  class_features = features["class_features"]
  audio = convert_to_feature([audio], "byte")
  rgb = convert_to_feature([rgb], "byte")
  class_features = convert_to_feature(class_features, "float")
  features = {"audio": tf.train.FeatureList(feature=[audio]), "rgb": tf.train.FeatureList(feature=[rgb]), "class_features": tf.train.FeatureList(feature=[class_features])}
  features = tf.train.FeatureLists(feature_list=features)
  return features

def serialize_combined_features(features):
  """Serialize features.

  Args:
    features: features of the video
  """
  audio = features["audio"][0].numpy().tostring()
  rgb = features["rgb"][0].numpy().tostring()
  print(features["class_features"])
  class_features = features["class_features"].tostring()
  audio = convert_to_feature([audio], "byte")
  rgb = convert_to_feature([rgb], "byte")
  class_features = convert_to_feature([class_features], "byte")
  print(tf.io.decode_raw(class_features, tf.float32))
  features = {"audio": tf.train.FeatureList(feature=[audio]), "rgb": tf.train.FeatureList(feature=[rgb]), "class_features": tf.train.FeatureList(feature=[class_features])}
  features = tf.train.FeatureLists(feature_list=features)
  return features

def serialize_video_context(context):
  """Serialize context for a video.

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
  context["candidate_labels"] = convert_to_feature(candidate_labels, "int")
  context = tf.train.Features(feature=context)
  return context

def serialize_segment_context(context, pipeline_type):
  """Serialize context for a segment.

  Args:
    context: context of the video
    pipeline_type: type of pipeline. Can be train or test
  """
  video_id = tf.convert_to_tensor(context["id"])[0]
  segment_label = context["segment_label"]
  segment_start_time = context["segment_start_time"]
  segment_score = context["segment_score"]
  if pipeline_type == "train":
    segment_label = convert_labels(segment_label)

  context["id"] = convert_to_feature([video_id.numpy()], "byte")
  context["segment_label"] = convert_to_feature(segment_label.numpy(), "int")
  context["segment_start_time"] = convert_to_feature(segment_start_time.numpy(), "int")
  context["segment_score"] = convert_to_feature(segment_score.numpy(), "float")
  if pipeline_type == "test":
    segment_id = context["segment_id"]
    candidate_label = context["candidate_label"]
    context["segment_id"] = convert_to_feature([segment_id],"int")
    context["candidate_label"] = convert_to_feature([candidate_label], "int")
  context = tf.train.Features(feature=context)
  return context

def serialize_class_segment_context(context, pipeline_type):
  """Serialize context for a segment from class feature generation.

  Args:
    context: context of the video
    pipeline_type: type of pipeline. Can be train or test
  """
  segment_label = context["segment_label"]
  segment_start_time = context["segment_start_time"]

  context["id"] = convert_to_feature([context["id"]], "byte")
  context["segment_label"] = convert_to_feature(segment_label.numpy(), "int")
  context["segment_start_time"] = convert_to_feature(segment_start_time.numpy(), "int")
  context["segment_score"] = convert_to_feature([context["segment_score"]], "float")
  if pipeline_type == "test":
    segment_id = context["segment_id"].numpy()
    candidate_label = context["candidate_label"].numpy()
    context["segment_id"] = convert_to_feature(segment_id,"int")
    context["candidate_label"] = convert_to_feature(candidate_label, "int")
  context = tf.train.Features(feature=context)
  return context

def serialize_combined_context(context):
  """Serialize context for a segment from class feature generation.

  Args:
    context: context of the video
  """
  context["id"] = convert_to_feature([context["id"]], "byte")
  context["segment_label"] = convert_to_feature(context["segment_label"].numpy(), "int")
  context = tf.train.Features(feature=context)
  return context

def serialize_data(context, features, type, pipeline_type="train"):
  """Serialize video or segment from context and features.

  Args:
    context: context of the video
    features: features of the video
    type: type of data to store. Can either be video, segment, or csf.
    pipeline_type: type of pipeline. Can be train or test
  """
  if type == "video":
    context = serialize_video_context(context)
    features = serialize_features(features)
  elif type == "segment":
    context = serialize_segment_context(context, pipeline_type)
    features = serialize_features(features)
  elif type == "csf":
    context = serialize_class_segment_context(context, pipeline_type)
    features = serialize_class_features(features)
  elif type == "combine_data":
    context = serialize_combined_context(context)
    features = serialize_combined_features(features)
  else:
    print("Incorrect type chosen for serialization.")
  example = tf.train.SequenceExample(feature_lists=features, context=context)
  return example.SerializeToString()

def save_shard(data_dir, shard, file_type, shard_number):
  """Save a shard data to data_dir as a TFRecords file.

  Args:
    data_dir: directory for file to be saved
    shard: list of serialized examples to be saved
    file_type: prefix of file name.
    shard_number: suffix of file name.
  """
  print(f"Processing shard number {shard_number}")
  shard = tf.convert_to_tensor(shard)
  shard_dataset = tf.data.Dataset.from_tensor_slices(shard)
  file_name = file_type + str(shard_number)
  file_path = os.path.join(data_dir, '%s.tfrecord' % file_name)
  writer = tf.data.experimental.TFRecordWriter(file_path)
  writer.write(shard_dataset)

def save_data(new_data_dir, input_dataset, candidates, file_type="test", shard_size=17):
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
    serialized_video = serialize_data(context, features, "video")
    shard.append(serialized_video)
    shard_counter += 1
    if shard_counter == shard_size:
      save_shard(new_data_dir, shard, file_type, shard_number)
      shard_counter = 0
      shard_number += 1
      shard = []
  #Handles overflow
  if shard_counter != 0:
    save_shard(new_data_dir, shard, file_type, shard_number)
    shard_counter = 0
    shard_number += 1
    shard = []

def split_data(data_dir, input_dataset, shard_size=85, num_classes=1000, file_type="class", pipeline_type="train"):
  """Save data as TFRecords Datasets in new_data_dir.

  Args:
    new_data_dir: string giving the directory to save TFRecords Files
    input_dataset: original dataset before candidate generation
  """
  #Context: id, label, score
  #Features: rgb, audio for 1 segment
  #Convert input_dataset from video level data to multiple segments.
  video_holder = [[] for i in range(num_classes)]
  video_number = 0
  number_faulty_examples = 0
  for video in input_dataset:
    print(f"Processing video number {video_number}")
    context = video[0]
    features = video[1]
    segment_start_times = context["segment_start_times"].values.numpy()
    for segment_index in range(len(segment_start_times)):
      if segment_start_times[segment_index]+5 <= tf.shape(features["rgb"])[1]:
        new_context, new_features = {}, {}
        segment_score = context["segment_scores"].values.numpy()[segment_index]
        new_context["id"] = context["id"]
        new_context["segment_label"] = tf.convert_to_tensor([context["segment_labels"].values.numpy()[segment_index]])
        new_context["segment_start_time"] = tf.convert_to_tensor([segment_start_times[segment_index]])
        new_context["segment_score"] = tf.convert_to_tensor([context["segment_scores"].values.numpy()[segment_index]])
        new_features["rgb"] = features["rgb"][:,segment_start_times[segment_index]:segment_start_times[segment_index]+5,:]
        new_features["audio"] = features["audio"][:,segment_start_times[segment_index]:segment_start_times[segment_index]+5,:]
        if pipeline_type == "train":
          label = new_context["segment_label"]
          label = convert_labels(label).numpy()[0]
          serialized_video = serialize_data(new_context, new_features, "segment", pipeline_type=pipeline_type)
          video_holder[label].append(serialized_video)
        elif pipeline_type == "test":
          if segment_score == 1:
            new_context["segment_id"] = np.array(segment_index)
            candidate_classes = context["candidate_labels"].values.numpy()
            for candidate_class in candidate_classes:
              new_context_copy = new_context.copy()
              new_features_copy = new_features.copy()
              new_context_copy["candidate_label"] = candidate_class
              serialized_video = serialize_data(new_context_copy, new_features_copy, "segment", pipeline_type=pipeline_type)
              video_holder[candidate_class].append(serialized_video)
      else:
        video_size = tf.shape(features["rgb"])[1]
        segment_time = segment_start_times[segment_index]
        print(f"Error, video not long enough {video_size} for segment start time {segment_time}")
        number_faulty_examples += 1
    video_number += 1
  for shard_number in range(len(video_holder)):
    print(f"Class number {shard_number} has {len(video_holder[shard_number])} segments")
    if len(video_holder[shard_number]) != 0:
      save_shard(data_dir, video_holder[shard_number], file_type, shard_number)
  print(f"Number of faulty examples {number_faulty_examples}")