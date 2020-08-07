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

Defines the training procedure. Run this python file to train a new model.
"""
import getopt
import model as model_lib
import os
import readers
import reader_utils
import sys
import tensorflow as tf

def train(data_dir, model_path, logs_dir, epochs=100, lr=0.0002, num_clusters=150, batch_size=20, fc_units=512):
  """Train the video classifier model.

  Args:
    data_dir: path to data directory. Must have train, validate, test as subdirectories containing the respective data
  Returns:
    model: trained video classifier
  """
  data_reader = reader_utils.get_reader(class_features=True)
  train_dataset = data_reader.get_dataset(data_dir, batch_size)

  video_input_shape = (batch_size, 5, 1024)
  audio_input_shape = (batch_size, 5, 128)
  input_shape = (5, 1152)
  second_input_shape = (3)

  #Compile and train model
  model_generator = model_lib.SegmentClassifier(num_clusters, video_input_shape, audio_input_shape, fc_units=fc_units, num_classes=data_reader.num_classes)
  model = model_generator.build_model(input_shape, second_input_shape, batch_size)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["binary_accuracy"])
  model.summary()
  #Implement callbacks
  tensor_board = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, update_freq=100)
  model.fit(train_dataset, epochs=epochs, callbacks=[tensor_board])
  model.save_weights(model_path)
  return model

if __name__ == "__main__":
  assert len(sys.argv) == 3, ("Incorrect number of arguments {}. Should be 2. Please consult the README.md for proper argument use.".format(len(sys.argv)-1))
  short_options = "i:m:l:"
  long_options = ["input_dir=", "model_path=", "logs_dir="]
  try:
    arguments, values = getopt.getopt(sys.argv[1:], short_options, long_options)
  except getopt.error as err:
    print(str(err))
    sys.exit(2)

  for current_argument, current_value in arguments:
    if current_argument in ("-i", "--input_dir"):
      input_dir = current_value
    elif current_argument in ("-m", "--model_path"):
      model_path = current_value
    elif current_argument in ("-l", "--logs_dir"):
      logs_dir = current_value
  model = train(input_dir, model_path, logs_dir)