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
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import tf.keras.metrics as metrics


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

	for batch in test_dataset:
		test_input = tf.convert_to_tensor(batch[0])
		test_labels = tf.convert_to_tensor(batch[1])

		curr_time = time.time()
		predictions = model.predict(test_input)
		print(f"Prediction speed {time.time() - curr_time}")

		loss_vals = loss.eval_loss(test_labels, predictions)

		#Update Metrics
		curr_time = time.time()
		auc_calculator.update_state(test_labels, predictions)
		pr_calculator.update_state(test_labels, predictions)
		rp_calculator.update_state(test_labels, predictions)

		print(f"Accumulate time {time.time() - curr_time}")

		print(f"Batch Number {batch_num} with loss {tf.math.reduce_mean(loss_vals)}.")
		batch_num += 1
	
	#Get results
	auc_pr = auc_calculator.result()
	precision = pr_calculator.result()
	recall = rp_calculator.result()

	eval_dict = {"AUCPR": auc_pr, "precision": precision, "recall": recall}

	return eval_dict