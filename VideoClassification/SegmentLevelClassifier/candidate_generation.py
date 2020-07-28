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

Used to perform candidate generation preprocessing step
"""
import candidate_generation_utils as utils
import model as model_lib
import readers
import tensorflow as tf
import tensorflow_datasets as tfds
import writer

def load_model(model_path, num_clusters=256, batch_size=80, random_frames=True, num_mixtures=2, fc_units=1024, iterations=300, num_classes=3862):
  """Load Video Classifier model."""
  video_input_shape = (batch_size, iterations, 1024)
  audio_input_shape = (batch_size, iterations, 128)
  input_shape = (iterations, 1152)
  model_generator = model_lib.VideoClassifier(num_clusters, video_input_shape, audio_input_shape, fc_units=fc_units, num_classes=num_classes, num_mixtures=num_mixtures, iterations=iterations)
  model = model_generator.build_model(input_shape, batch_size)
  model.load_weights(model_path)
  return model

def generate_candidates(input_dataset, model, k, class_csv):
  """Generate top k candidates per class.

  Args:
    input_dataset: dataset to be chosen from
    model: model used to rank data
    k: number of candidates per class
    class_csv: path to csv file containing the indices of the classes used, out of the 3.8k output classes from the video-level classifier.
  Returns:
    candidates: list of lists where each inner list contains the class indices that the corresponding input data is a candidate for. len(candidates) == len(input_dataset)
  """
  probability_holder = utils.PROBABILITY_HOLDER(class_csv, k)
  video_num = 0
  input_dataset = tfds.as_numpy(input_dataset)
  for video in input_dataset:
    print(f"Processing video number {video_num}")
    video_id = tf.convert_to_tensor(video[0])[0].numpy()
    video_input = tf.convert_to_tensor(video[1])
    probability_holder.add_data(video_id, model.predict(video_input)[0])
    video_num += 1
  return probability_holder.find_candidates()

if __name__ == "__main__":
  video_reader = readers.VideoDataset()
  input_dataset = video_reader.get_dataset("/home/conorfvedova_google_com/data/segments/test", batch_size=1, type="test")
  model = load_model("../model_weights.h5")
  candidates = generate_candidates(input_dataset, model, 200, "vocabulary.csv")
  segment_reader = readers.PreprocessingDataset()
  input_dataset = segment_reader.get_dataset("/home/conorfvedova_google_com/data/segments/test", batch_size=1, type="test")
  writer.save_data("/home/conorfvedova_google_com/data/segments/candidate_test", input_dataset, candidates)
