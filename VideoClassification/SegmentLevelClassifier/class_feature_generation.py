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
import time

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

def compute_and_save(data_dir, input_dir, num_classes=1000):
  """Compute class specific features for input_dataset and save them to data_dir.

  Args:
    data_dir: directory to save data to
    input_dir: directory where input data is stored.
  """
  #Need to take into account segment_weights
  #Store previous computations to speed up runtime
  num_segment = 0
  for label in range(num_classes):
    start_time = time.time()
    shard = []
    input_dataset_reader = readers.SegmentDataset(class_num=label)
    input_dataset = input_dataset_reader.get_dataset("/home/conorfvedova_google_com/data/segments/split_validation", batch_size=1, type="class")
    first_of_class = True
    #Preload Data
    video_holder = []
    for segment in input_dataset:
      context = segment[0]
      features = segment[1]
      video_holder.append((context, features))
    for segment in video_holder:
      print(f"Processing segment {num_segment}")
      context = segment[0]
      features = segment[1]
      video_id = tf.convert_to_tensor(context["id"])[0].numpy()
      total_positive = 0
      total_negative = 0
      #if first_of_class:
      calculation_time = time.time()
      for comparison_segment in video_holder:
        real_calc_time = time.time()
        comparison_context = comparison_segment[0]
        comparison_features = comparison_segment[1]
        #video_holder.append((comparison_context, comparison_features))
        comparison_video_id = tf.convert_to_tensor(comparison_context["id"])[0].numpy()
        if video_id == comparison_video_id:
          positive, negative = 0,0
        else:
          segment_score = comparison_context["segment_score"][0].numpy()
          positive, negative = 0,0
          if segment_score == 0:
            realer_time = time.time()
            negative = calculate_cosine(features["rgb"][0].numpy(), comparison_features["rgb"][0].numpy())
            negative += calculate_cosine(features["audio"][0].numpy(), comparison_features["audio"][0].numpy())
            print(f"Negative Calculation time {time.time() - realer_time}")
          else:
            post_time = time.time()
            positive = calculate_cosine(features["rgb"][0].numpy(), comparison_features["rgb"][0].numpy())
            positive += calculate_cosine(features["audio"][0].numpy(), comparison_features["audio"][0].numpy())
            print(f"Positive Calculation time {time.time() - post_time}")
          total_positive += positive
          total_negative += negative
        print(f"Real time {time.time() - real_calc_time}")
      first_of_class = False
      print(f"Calculation time {time.time() - calculation_time}")
      assert False
      # else:
      #   for comparison_segment_index in range(len(video_holder)):
      #     comparison_segment = video_holder[comparison_segment_index]
      #     comparison_context = comparison_segment[0]
      #     comparison_features = comparison_segment[1]
      #     comparison_video_id = tf.convert_to_tensor(comparison_context["id"])[0].numpy()
      #     if video_id == comparison_video_id:
      #       positive, negative = 0,0
      #     else:
      #       segment_score = comparison_context["segment_score"][0].numpy()
      #       positive, negative = 0,0
      #       if segment_score == 0:
      #         negative = calculate_cosine(features["rgb"][0].numpy(), comparison_features["rgb"][0].numpy())
      #         negative += calculate_cosine(features["audio"][0].numpy(), comparison_features["audio"][0].numpy())
      #       else:
      #         positive = calculate_cosine(features["rgb"][0].numpy(), comparison_features["rgb"][0].numpy())
      #         positive += calculate_cosine(features["audio"][0].numpy(), comparison_features["audio"][0].numpy())
      #       total_positive += positive
      #       total_negative += negative
      serialization_time = time.time()
      features["class_features"] = tf.convert_to_tensor([total_positive, total_negative])
      shard.append(writer.serialize_data(context.copy(), features.copy(), "csf"))
      num_segment += 1
      print(f"Serialization time {time.time() - serialization_time}")

      if total_negative == 0 or total_positive == 0:
        print(f"Invalid calculation for segment {num_segment-1}")
        assert False
    writer.save_shard(data_dir, shard, "train", label)
    print(time.time() - start_time)
    assert False

if __name__ == "__main__":
  compute_and_save("/home/conorfvedova_google_com/data/segments/input_train_data", "/home/conorfvedova_google_com/data/segments/split_validation")