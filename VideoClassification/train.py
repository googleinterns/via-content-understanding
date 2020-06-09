import tensorflow as tf
import NetVLAD_CG
import reader_utils
import loss
import tensorflow_datasets as tfds

import eval_util

import readers

def check_in(i, temp_list):
	for j in temp_list:
		if (i == j).all():
			return True
		else:
			return False

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

	train_dataset = data_reader.get_dataset(train_dir, batch_size=batch_size, num_epochs=num_epochs)

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
	test_dataset = data_reader.get_dataset(test_dir, batch_size=batch_size, type="train")
	test_dataset = tfds.as_numpy(test_dataset)
	evaluation_metrics = eval_util.EvaluationMetrics(data_reader.num_classes, 20)
	for batch in numpy_dataset:
		test_input = tf.convert_to_tensor(batch[0])
		test_frames = tf.convert_to_tensor(batch[1])
		test_labels = tf.convert_to_tensor(batch[2])

		predictions = model.predict(x=[test_input, test_frames])
		
		loss_vals = loss.eval_loss(test_labels, predictions)

		test_labels = test_labels.numpy()
		loss_vals = loss_vals.numpy()

		evaluation_metrics.accumulate(predictions, test_labels, loss_vals)
	eval_dict = evaluation_metrics.get()

	print(eval_dict)

	return eval_dict

def train(epochs=15, lr=0.01, num_clusters=256, batch_size=64, random_frames=True, num_mixtures=2, fc_units=1024, iterations=300):
	#Set up Reader and Preprocess Data
	data_reader, train_dataset, validation_dataset = load_datasets('/home/conorfvedova_google_com/data/train/', '/home/conorfvedova_google_com/data/validate/', epochs, batch_size)

	iterator = tfds.as_numpy(train_dataset)
	temp_list = []
	for i in iterator:
		if check_in(i[0], temp_list):
			print(f"Not in list, len: {len(temp_list)}\n")
			assert False
		else:
			temp_list.append(i[0])
			print(len(temp_list))


	assert False

	video_input_shape = (batch_size, iterations, 1024)
	audio_input_shape = (batch_size, iterations, 128)
	input_shape = (batch_size, iterations, 1152)

	#Compile and train model
	model = NetVLAD_CG.VideoClassifier(num_clusters, video_input_shape, audio_input_shape, fc_units=fc_units, num_classes=data_reader.num_classes, num_mixtures=num_mixtures, iterations=iterations, random_frames=random_frames)
	
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss.custom_crossentropy, metrics=['categorical_accuracy'])
	model.build(input_shape)
	model.summary()
	batch_counter = 0
	for batch in train_dataset:
		train_input = tf.convert_to_tensor(batch[0])
		train_frames = tf.convert_to_tensor(batch[1])
		train_labels = tf.convert_to_tensor(batch[2])

		model.train_on_batch(x=[train_input, train_frames], y=train_labels)

		batch_counter += 1

	#Evaluate model
	test_model(model, data_reader, '/home/conorfvedova_google_com/data/train/', batch_size)

	#model.save_weights('/home/conorfvedova_google_com/saved_model/model-final.h5')

if __name__ == "__main__":
	train()
