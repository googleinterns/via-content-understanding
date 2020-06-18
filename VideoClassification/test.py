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

Defines the testing procedure.
"""
import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.metrics as metrics
import numpy as np

import readers
import reader_utils
import model as model_lib
import loss

def test_model(model, data_reader, test_dir, batch_size):
	"""Test the model on test dataset attained from test_dir.
	
	Args:
		model: the trained keras model
		data_reader: the TfRecords datareader used to load training and validation datasets
		test_dir: string representing the directory containing the test TfRecords files
	Returns:
		eval_dict: dictionary containing important evaulation metrics
	"""
	#Create evaluation metrics
	auc_calculator = metrics.AUC(multi_label=True, curve='PR')
	pr_calculator = metrics.PrecisionAtRecall(0.7)
	rp_calculator = metrics.RecallAtPrecision(0.7)

	#Prepare data
	batch_num = 0
	test_dataset = data_reader.get_dataset(test_dir, batch_size=batch_size, type="test")
	test_dataset = tfds.as_numpy(test_dataset)

	num_samples = 0
	total_true = 0
	for batch in test_dataset:
		test_input = tf.convert_to_tensor(batch[0])
		test_labels = tf.convert_to_tensor(batch[1])

		predictions = model.predict(test_input)

		#loss_val = loss.custom_crossentropy(test_labels, predictions)

		#Update Metrics
		# auc_calculator.update_state(test_labels, predictions)
		# pr_calculator.update_state(test_labels, predictions)
		# rp_calculator.update_state(test_labels, predictions)

		# auc_pr = auc_calculator.result()
		# precision = pr_calculator.result()
		# recall = rp_calculator.result()

		# eval_dict = {"AUCPR": auc_pr, "precision": precision, "recall": recall}

		#print(eval_dict)

		num_true = np.sum(test_labels[np.argmax(predictions, 1)])
		total_true += num_true
		num_samples += 80
		print(f"Batch Number {batch_num} with num_true {num_true / 80}.")
		batch_num += 1
	
	#Get results
	# auc_pr = auc_calculator.result()
	# precision = pr_calculator.result()
	# recall = rp_calculator.result()

	# eval_dict = {"AUCPR": auc_pr, "precision": precision, "recall": recall}

	eval_duct = {"prec": total_true / num_samples}

	return eval_dict

def load_and_test(data_dir, model_path, epochs=6, lr=0.0002, num_clusters=256, batch_size=80, random_frames=True, num_mixtures=2, fc_units=1024, iterations=300):
	"""Load and test the video classifier model.

	Args:
		data_dir: path to data directory. Must have test as a subdirectory containing the respective data
		model_path: path to the model weights save file. Must be a .h5 file
	"""
	test_dir = os.path.join(data_dir, "test")

	#Set up Reader and Preprocess Data
	data_reader = reader_utils.get_reader()

	video_input_shape = (batch_size, iterations, 1024)
	audio_input_shape = (batch_size, iterations, 128)
	input_shape = (iterations, 1152)

	#Compile and train model
	model_generator = model_lib.VideoClassifier(num_clusters, video_input_shape, audio_input_shape, fc_units=fc_units, num_classes=data_reader.num_classes, num_mixtures=num_mixtures, iterations=iterations)
	
	model = model_generator.build_model(input_shape, batch_size)

	model.load_weights(model_path)

	eval_dict = test_model(model, data_reader, test_dir, batch_size)
	print(eval_dict)

if __name__ == "__main__":
	load_and_test("/home/conorfvedova_google_com/data", "model_weights.h5")