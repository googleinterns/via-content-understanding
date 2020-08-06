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

Compute and add Class specific features to the data.
"""
import numpy as np
import readers
import tensorflow as tf
import writer

def calculate_cosine(segment1, segment2):
  """Calculate the cosine of the angle between segment1 and segment2. Modified to work with matrices by applying mean of cosine similarities.

  Args:
    segment1: Matrix of vectors to compare
    segment2: Matrix of vectors to compare
  """
  similarity = []
  for i in range(len(segment1)):
    similarity.append(np.dot(segment1[i], segment2[i]) / (np.linalg.norm(segment1[i])*np.linalg.norm(segment2[i])))
  similarity = np.array(similarity)
  return np.mean(similarity)

def compute_and_save(data_dir, input_dir, comparison_directory="/home/conorfvedova_google_com/data/segments/split_validation", pipeline_type="test", num_classes=1000):
  """Compute class specific features for input_dataset and save them to data_dir.

  Args:
    data_dir: directory to save data to
    input_dir: directory where input data is stored.
  """
  num_segment = 0
  #for label in range(num_classes):
  shard = []
  label = 999
  comparison_dataset_reader = readers.SegmentDataset(class_num=label)
  comparison_dataset = comparison_dataset_reader.get_dataset(comparison_directory, batch_size=1, type="class")
  
  #Preload Data and convert to numpy for calculations
  video_holder_input = []
  video_holder_comparison = []
  for segment in comparison_dataset:
      context = segment[0]
      features = segment[1]
      features["rgb"] = features["rgb"][0].numpy()
      features["audio"] = features["audio"][0].numpy()
      context["id"] = tf.convert_to_tensor(context["id"])[0].numpy()
      context["segment_score"] = context["segment_score"][0].numpy()
      video_holder_comparison.append((context, features))
  if pipeline_type == "test":
    input_dataset_reader = readers.SegmentDataset(class_num=label, pipeline_type="train")
    input_dataset = input_dataset_reader.get_dataset(input_dir, batch_size=1, type="class")
    for segment in input_dataset:
      context = segment[0]
      features = segment[1]
      features["rgb"] = features["rgb"][0].numpy()
      features["audio"] = features["audio"][0].numpy()
      context["id"] = tf.convert_to_tensor(context["id"])[0].numpy()
      context["segment_score"] = context["segment_score"][0].numpy()
      video_holder_input.append((context, features))
  else:
    video_holder_input = video_holder_comparison
  for segment in video_holder_input:
    print(f"Processing segment {num_segment}")
    context = segment[0]
    features = segment[1]
    video_id = context["id"]
    total_positive = 0
    total_negative = 0
    for comparison_segment in video_holder_comparison:
      comparison_context = comparison_segment[0]
      comparison_features = comparison_segment[1]
      comparison_video_id = comparison_context["id"]
      if video_id == comparison_video_id:
        positive, negative = 0,0
      else:
        segment_score = comparison_context["segment_score"]
        positive, negative = 0,0
        if segment_score == 0:
          negative = calculate_cosine(features["rgb"], comparison_features["rgb"])
          negative += calculate_cosine(features["audio"], comparison_features["audio"])
        else:
          positive = calculate_cosine(features["rgb"], comparison_features["rgb"])
          positive += calculate_cosine(features["audio"], comparison_features["audio"])
        total_positive += positive
        total_negative += negative
    features["class_features"] = np.array([total_positive, total_negative])
    shard.append(writer.serialize_data(context.copy(), features.copy(), "csf", pipeline_type="train"))
    num_segment += 1
    if total_negative == 0 or total_positive == 0:
      print(f"Invalid calculation for segment {num_segment-1}")
      assert False
  writer.save_shard(data_dir, shard, "train", 1000)

if __name__ == "__main__":
  compute_and_save("/home/conorfvedova_google_com/data/segments/input_train_data", "/home/conorfvedova_google_com/data/segments/split_train2", pipeline_type="train")