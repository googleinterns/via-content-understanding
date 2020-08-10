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
import getopt
import model as model_lib
import readers
import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import writer

def load_model(model_path, num_clusters=256, batch_size=80, random_frames=True, num_mixtures=2, fc_units=1024, num_input_frames=300, num_classes=3862):
  """Load video classifier model."""
  video_input_shape = (batch_size, num_input_frames, 1024)
  audio_input_shape = (batch_size, num_input_frames, 128)
  input_shape = (num_input_frames, 1152)
  model_generator = model_lib.VideoClassifier(num_clusters, video_input_shape, audio_input_shape, fc_units=fc_units, num_classes=num_classes, num_mixtures=num_mixtures, iterations=num_input_frames)
  model = model_generator.build_model(input_shape, batch_size)
  model.load_weights(model_path)
  return model

def generate_candidates(input_dataset, model, k, class_csv):
  """Generate top k candidates per class.

  Args:
    input_dataset: tf.dataset to be chosen from. Dataset must be attained from VideoClassifier from readers.py
    model: model used to rank data
    k: number of candidates per class
    class_csv: path to csv file containing the indices of the classes used, out of the 3.8k output classes from the video-level classifier.
    The integers denoting chosen classes should be in the first column.
  Returns:
    candidates: list of lists where each inner list contains the class indices that the corresponding input data is a candidate for. len(candidates) == len(input_dataset)
  """
  probability_holder = utils.ProbabilityHolder(class_csv, k)
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
  assert len(sys.argv) == 5, ("Incorrect number of arguments {}. Should be 4. Please consult the README.md for proper argument use.".format(len(sys.argv)-1))
  short_options = "i:m:f:w:"
  long_options = ["input_dir=", "model_weights_path=", "file_type_name=", "write_dir="]
  try:
    arguments, values = getopt.getopt(sys.argv[1:], short_options, long_options)
  except getopt.error as err:
    print(str(err))
    sys.exit(2)

  for current_argument, current_value in arguments:
    if current_argument in ("-i", "--input_dir"):
      input_dir = current_value
    elif current_argument in ("-m", "--model_weights_path"):
      model_weights_path = current_value
    elif current_argument in ("-f", "--file_type_name"):
      file_type_name = current_value
    elif current_argument in ("-w", "--write_dir"):
      write_dir = current_value

  video_reader = readers.VideoDataset()
  input_dataset = video_reader.get_dataset(input_dir, batch_size=1, type=file_type_name)
  model = load_model(model_weights_path)
  candidates = generate_candidates(input_dataset, model, 50, "vocabulary.csv")
  segment_reader = readers.BasicDataset()
  input_dataset = segment_reader.get_dataset(input_dir, batch_size=1, type=file_type_name)
  writer.save_data(write_dir, input_dataset, candidates)