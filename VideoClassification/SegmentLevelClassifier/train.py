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
import model as model_lib
import os
import readers
import reader_utils
import tensorflow as tf

def crossentropy(y_actual, y_predicted, epsilon=10e-6, alpha=0.5):
  """Custom Crossentropy function used for stability.

  Args:
    y_actual: actual output labels
    y_predicted: predicted output label probabilities
    epsilon: Small value to ensure log is not taken of 0. 10e-6 is default.
    alpha: Magnifies the contribution of correctly predicting as opposed to incorrectly predicting. Alpha of 0.5 is default scaling and does not modify the loss.

  Returns:
    tensor giving the loss for y_actual versus y_predicted.
  """
  float_labels = tf.cast(y_actual, tf.float32)

  print(float_labels)
  print(y_predicted)
  cross_entropy_loss = 2*(alpha*float_labels * tf.math.log(y_predicted + epsilon) + (1-alpha)*(1 - float_labels) * tf.math.log(1 - y_predicted + epsilon))
  cross_entropy_loss = tf.math.negative(cross_entropy_loss)

  return tf.math.reduce_mean(tf.math.reduce_sum(cross_entropy_loss, 1))


def train(data_dir, epochs=6, lr=0.0002, num_clusters=10, batch_size=20, fc_units=512):
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
  tensor_board = tf.keras.callbacks.TensorBoard(log_dir="logs2", update_freq=100)
  model.fit(train_dataset, epochs=epochs, callbacks=[tensor_board])

  model.save_weights("model_weights_segment_level.h5")

  #Evaluate model
  eval_dict = evaluate.evaluate_model(model, data_reader, test_dir, batch_size)

  print(eval_dict)

  return model

if __name__ == "__main__":
  model = train("/home/conorfvedova_google_com/data/segments/input_train_data")