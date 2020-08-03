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
import readers

#Reader has ((video_matrix, class_features_list), label). Batch it? Nah lets just go 1 at a time. Less efficient but easier to write.

def evaluate_example(model, example, num_classes=1000):
  """Evaluate one example using model.
  
  Args
    model: A keras model with input (video_matrix, class_feature_list) and 1 output denoting class relevance
    example: One example with the following format: (video_matrix, list of class_features_list with len == number of candidate classes)
    label is one-hot encoding
  """
  predictions = [0] * num_classes
  video_matrix = tf.convert_to_tensor(example[0])
  class_features_lists = tf.convert_to_tensor(example[1])
  for class_features_list in class_features_lists:
    prediction = model.predict((video_matrix, class_features_list))
    class_num = class_features_list[0].numpy()
    predictions[class_num] = prediction
  return predictions

def evaluate_model(model, dataset):
  """Evaluate the model and dataset.

  Args:
   model: A keras model with input (video_matrix, class_feature_list) and 1 output denoting class relevance
   dataset: tf.dataset attained from a Dataset class from readers.py  
  """
  #Batch size of 1
  #input_data is list of inputs
  #label is a one-hot
  #Create evaluation metrics
  aucroc_calculator = metrics.AUC(multi_label=True)
  aucpr_calculator = metrics.AUC(multi_label=True, curve='PR')
  pr_calculator = metrics.PrecisionAtRecall(0.7)
  rp_calculator = metrics.RecallAtPrecision(0.7)
  for input_data, label in dataset:
    prediction = evaluate_example(model, input_data)
    #Update Metrics
    aucroc_calculator.update_state(test_labels, predictions)
    aucpr_calculator.update_state(test_labels, predictions)
    pr_calculator.update_state(test_labels, predictions)
    rp_calculator.update_state(test_labels, predictions)
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
  second_input_shape = (3)

  model_generator = model_lib.SegmentClassifier(num_clusters, video_input_shape, audio_input_shape, fc_units=fc_units, num_classes=data_reader.num_classes)
  model = model_generator.build_model(input_shape, second_input_shape, batch_size)
  model.load_weights(model_path)
  eval_dict = evaluate_model(model, test_dataset)
  print(eval_dict)

if __name__ == "__main__":
  load_and_evaluate("/home/conorfvedova_google_com/data", "model_weights.h5")