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

Evaluate different performance metrics of given model.
"""
import model as model_lib
import os
import readers
import tensorflow as tf
import tensorflow.keras.metrics as metrics
import numpy as np

def evaluate_example(model, example, num_classes=1000):
  """Evaluate one example using model.
  
  Args
    model: A keras model with input (video_matrix, class_feature_list) and 1 output denoting class relevance
    example: One example with the following format: (video_matrix, list of class_features_list with len == number of candidate classes)
    label is one-hot encoding
  """
  predictions = [0] * num_classes
  video_matrix = tf.convert_to_tensor(example[0])
  class_features_lists = tf.reshape(example[1].values, [-1, 1, 3])
  for class_features_list in class_features_lists:
    prediction = model.predict((video_matrix, class_features_list))
    class_num = tf.cast(class_features_list[0][0], tf.int64).numpy()
    predictions[class_num] = prediction[0][0]
  return tf.reshape(tf.convert_to_tensor(predictions), [1,-1])

def evaluate_model(model, dataset):
  """Evaluate the model and dataset.

  Args:
   model: A keras model with input (video_matrix, class_feature_list) and 1 output denoting class relevance
   dataset: tf.dataset attained from a Dataset class from readers.py  
  """
  segment_num = 0
  #Create evaluation metrics
  aucroc_calculator = metrics.AUC()
  aucpr_calculator = metrics.AUC(curve='PR')
  pr_calculator = metrics.PrecisionAtRecall(0.7)
  rp_calculator = metrics.RecallAtPrecision(0.7)
  for input_data, label in dataset:
    prediction = evaluate_example(model, input_data)
    #Update Metrics
    aucroc_calculator.update_state(label, prediction)
    aucpr_calculator.update_state(label, prediction)
    pr_calculator.update_state(label, prediction)
    rp_calculator.update_state(label, prediction)
    print(f"Processing segment number {segment_num}")
    segment_num += 1
  #Get results
  auc_roc = aucroc_calculator.result()
  auc_pr = aucpr_calculator.result()
  precision = pr_calculator.result()
  recall = rp_calculator.result()
  eval_dict = {"AUCPR": auc_pr, "AUCROC": auc_roc, "precision": precision, "recall": recall}
  return eval_dict

def load_and_evaluate(data_dir, model_path, num_clusters=10, batch_size=20, fc_units=512):
  """Load and test the video classifier model.

  Args:
    data_dir: path to data directory. Must have test as a subdirectory containing the respective data
    model_path: path to the model weights save file. Must be a .h5 file
  """
  data_reader = readers.EvaluationDataset()
  test_dataset = data_reader.get_dataset(data_dir, batch_size)

  video_input_shape = (batch_size, 5, 1024)
  audio_input_shape = (batch_size, 5, 128)
  input_shape = (5, 1152)
  second_input_shape = (1)

  model_generator = model_lib.SegmentClassifier(num_clusters, video_input_shape, audio_input_shape, fc_units=fc_units, num_classes=data_reader.num_classes)
  model = model_generator.build_model(input_shape, second_input_shape, batch_size)
  model.load_weights(model_path)
  eval_dict = evaluate_model(model, test_dataset)
  print(eval_dict)

if __name__ == "__main__":
  load_and_evaluate("/home/conorfvedova_google_com/data/segments/finalized_test_data", "model_weights_segment_level_baseline.h5")
