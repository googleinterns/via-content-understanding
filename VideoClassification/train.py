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
import os
import tensorflow as tf
import NetVLAD_CG
import reader_utils
import loss
import tensorflow_datasets as tfds

import eval_util

import readers

def load_datasets(train_dir, validate_dir, num_epochs, batch_size):
	"""Set up data reader and load training and validation datasets
	
	Args:
		train_dir: string representing the directory containing the train TfRecords files
		validate_dir: string representing the directory containing the validate TfRecords files
	Returns:
		data_reader: the TfRecords datareader. Can be reused to load other datasets
		train_dataset: training dataset after parsing
		validation_dataset: validation dataset after parsing
	"""
	data_reader = reader_utils.get_reader()

	train_dataset = data_reader.get_dataset(train_dir, batch_size=batch_size)

	validation_dataset = data_reader.get_dataset(validate_dir, batch_size=batch_size, type="validate")

	return data_reader, train_dataset, validation_dataset

def test_model(model, data_reader, test_dir, batch_size):
	"""Test the model on test dataset attained from test_dir.
	
	Args:
		model: the trained keras model
		data_reader: the TfRecords datareader used to load training and validation datasets
		test_dir: string representing the directory containing the test TfRecords files
	Returns:
		eval_dict: dictionary containing important evaulation metrics
	"""
	batch_num = 0
	test_dataset = data_reader.get_dataset(test_dir, batch_size=batch_size, type="test")

	test_dataset = tfds.as_numpy(test_dataset)
	evaluation_metrics = eval_util.EvaluationMetrics(data_reader.num_classes, 20)
	for batch in test_dataset:
		test_input = tf.convert_to_tensor(batch[0])
		test_labels = tf.convert_to_tensor(batch[1])

		predictions = model.predict(test_input)
		
		loss_vals = loss.eval_loss(test_labels, predictions)

		test_labels = test_labels.numpy()
		loss_vals = loss_vals.numpy()

		evaluation_metrics.accumulate(predictions, test_labels, loss_vals)
		batch_num += 1
		print(f"Batch Number {batch_num} with loss {tf.math.reduce_mean(loss_vals)}.")
	eval_dict = evaluation_metrics.get()

	print(eval_dict)

	return eval_dict

def train(data_dir, epochs=6, lr=0.0002, num_clusters=256, batch_size=80, random_frames=True, num_mixtures=2, fc_units=1024, iterations=300):
	"""Train the video classifier model.

	Args:
		data_dir: path to data directory. Must have train, validate, test as subdirectories containing the respective data
	Returns:
		model: trained video classifier

	"""
	train_dir = os.path.join(data_dir, "train")
	validation_dir = os.path.join(data_dir, "validate")
	test_dir = os.path.join(data_dir, "test")

	#Set up Reader and Preprocess Data
	data_reader, train_dataset, validation_dataset = load_datasets(train_dir, validation_dir, epochs, batch_size)

	video_input_shape = (batch_size, iterations, 1024)
	audio_input_shape = (batch_size, iterations, 128)
	input_shape = (iterations, 1152)
	frames_input_shape = ()

	#Compile and train model
	model_generator = NetVLAD_CG.VideoClassifier(num_clusters, video_input_shape, audio_input_shape, fc_units=fc_units, num_classes=data_reader.num_classes, num_mixtures=num_mixtures, iterations=iterations, random_frames=random_frames)
	
	model = model_generator.build_model(input_shape, frames_input_shape, batch_size)

	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss.custom_crossentropy, metrics=[tf.keras.metrics.Precision()])

	model.summary()
	
	#Implement callbacks
	#tensor_board = tf.keras.callbacks.TensorBoard(log_dir="logs2", update_freq=100)

	model.fit(train_dataset, epochs=epochs)#, validation_data=validation_dataset)#, callbacks=[tensor_board])

	#Evaluate model
	eval_dict = test_model(model, data_reader, test_dir, batch_size)

	return model

if __name__ == "__main__":
	train("/home/conorfvedova_google_com/data")
