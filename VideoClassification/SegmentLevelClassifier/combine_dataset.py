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

Combine segment records back together so that they may be easily evaluated. At this point, segments have been split according to their
candidate labels, but they need to be put back together.
"""
import numpy as np
import readers
import tensorflow as tf
import writer

def combine_data(data_dir, input_dir, shard_size=85, file_type="test"):
  """Read over all data and save the generated class specific features.
  
  Args:
    data_dir: directory to save data to
    input_dir: directory where input data is stored.
  """
  feature_storage = {}
  input_dataset_reader = readers.CombineSegmentDataset()
  input_dataset = input_dataset_reader.get_dataset(input_dir, batch_size=1)
  for segment in input_dataset:
    context = segment[0]
    features = segment[1]
    video_id = tf.convert_to_tensor(context["id"])[0].numpy()
    segment_id = context["segment_id"][0].numpy()
    candidate_label = tf.cast(context["candidate_label"], tf.float32).numpy()
    class_features = features["class_features"][0][0].numpy()
    class_features = np.array(candidate_label.tolist() + class_features.tolist())
    #Since the number of candidate classes is unknown, we must extend the storage list as we go.
    if video_id in feature_storage.keys():
      current_list = feature_storage[video_id]
      if segment_id >= len(current_list):
        extension_size = segment_id+1 - len(current_list)
        extension_list = [[] for i in range(extension_size)]
        extension_list[-1].append(class_features)
        feature_storage[video_id] = current_list + extension_list
      else:
        current_list[segment_id].append(class_features)
    else:
      extension_list = [[] for i in range(segment_id+1)]
      extension_list[-1].append(class_features)
      feature_storage[video_id] = extension_list
  
  #Store data
  save_dataset_reader = readers.CombineSegmentDataset()
  save_dataset = save_dataset_reader.get_dataset(input_dir, batch_size=1)
  stored_video_ids = {}
  segment_num, shard_number = 0,0
  shard = []
  for segment in save_dataset:
    context = segment[0]
    features = segment[1]
    video_id = tf.convert_to_tensor(context["id"])[0].numpy()
    segment_id = context["segment_id"][0].numpy()
    if video_id in stored_video_ids.keys():
      if segment_id not in stored_video_ids[video_id]:
        stored_video_ids[video_id].append(segment_id)
        new_context = {}
        new_context["id"] = tf.convert_to_tensor(context["id"])[0].numpy()
        new_context["segment_label"] = context["segment_label"]
        new_features = {}
        new_features["rgb"] = features["rgb"]
        new_features["audio"] = features["audio"]
        class_features_temp = feature_storage[video_id][segment_id]
        class_features_temp_add = []
        for i in class_features_temp:
          if i != []:
            class_features_temp_add.append(i)
        new_features["class_features"] = np.array(class_features_temp_add)
        shard.append(writer.serialize_data(new_context, new_features, "combine_data", pipeline_type="test"))
        segment_num += 1
    else:  
      stored_video_ids[video_id] = [segment_id]
      new_context = {}
      new_context["id"] = tf.convert_to_tensor(context["id"])[0].numpy()
      new_context["segment_label"] = context["segment_label"]
      new_features = {}
      new_features["rgb"] = features["rgb"]
      new_features["audio"] = features["audio"]
      class_features_temp = feature_storage[video_id][segment_id]
      class_features_temp_add = []
      for i in class_features_temp:
        if i != []:
          class_features_temp_add.append(i)
      new_features["class_features"] = np.array(class_features_temp_add)
      shard.append(writer.serialize_data(new_context, new_features, "combine_data", pipeline_type="test"))
      segment_num += 1
    if segment_num == shard_size:
      segment_num = 0
      writer.save_shard(data_dir, shard, file_type, shard_number)
      shard_number += 1
      shard = []
  if segment_num != 0:
      segment_num = 0
      writer.save_shard(data_dir, shard, file_type, shard_number)
      shard_number += 1
      shard = []

if __name__ == "__main__":
  combine_data("/home/conorfvedova_google_com/data/segments/finalized_test_data", "/home/conorfvedova_google_com/data/segments/input_test_data")