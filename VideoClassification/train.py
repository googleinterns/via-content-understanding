import tensorflow as tf
import NetVLAD_CG
import reader_utils
import loss
import tensorflow_datasets as tfds

import eval_util

import readers

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
NUM_EXAMPLES = 3844
NUM_VAL_EXAMPLES = 3844

def check_in(i, temp_list):
	for j in temp_list:
		if i == j:
			return True
		else:
			return False

if __name__ == "__main__":
	# Dataset flags.
	flags.DEFINE_string("train_dir", "~/data/train/train*.tfreco",
											"The directory to save the model files in.")
	flags.DEFINE_string(
			"train_data_pattern", "~/data/train/train*.tfrecord",
			"File glob for the training dataset. If the files refer to Frame Level "
			"features (i.e. tensorflow.SequenceExample), then set --reader_type "
			"format. The (Sequence)Examples are expected to have 'rgb' byte array "
			"sequence feature as well as a 'labels' int64 context feature.")
	flags.DEFINE_integer("num_readers", 8,
											 "How many threads to use for reading input files.")


def train(epochs=50, lr=0.01, num_clusters=256, batch_size=64, random_frames=True, num_mixtures=2, fc_units=1024, num_frames=30):
	steps_per_epoch = NUM_EXAMPLES // batch_size
	validation_steps = NUM_VAL_EXAMPLES // batch_size
				
	#Set up Reader and Preprocess Data
	data_reader = reader_utils.get_reader(num_samples=num_frames, random_frames=random_frames)

	train_dataset = data_reader.get_dataset('/home/conorfvedova_google_com/data/train/', batch_size=batch_size, num_workers=8)

	validation_dataset = data_reader.get_dataset('/home/conorfvedova_google_com/data/validate/', batch_size=batch_size, num_workers=8, type="validate")

	iterator = tfds.as_numpy(train_dataset)
	temp_list = []
	for i in iterator:
		if check_in(i, temp_list):
			print(f"Not in list, len: {len(temp_list)}\n")
		else:
			temp_list.append(i[0])
			print(temp_list)

	assert False

	video_input_shape = (batch_size, num_frames, 1024)
	audio_input_shape = (batch_size, num_frames, 128)
	input_shape = (batch_size, num_frames, 1152)

	#Compile and train model
	model = NetVLAD_CG.VideoClassifier(num_clusters, video_input_shape, audio_input_shape, fc_units=fc_units, num_classes=data_reader.num_classes, num_mixtures=num_mixtures)
	
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss.custom_crossentropy, metrics=['categorical_accuracy'])
	model.build(input_shape)
	model.summary()
	model.fit(x=train_dataset, steps_per_epoch=steps_per_epoch, validation_data=validation_dataset, validation_steps=validation_steps, epochs=epochs)

	#Evaluate model
	test_dataset = data_reader.get_dataset('/home/conorfvedova_google_com/data/train/', batch_size=batch_size, num_workers=8, type="train")
	numpy_dataset = tfds.as_numpy(test_dataset)
	evaluation_metrics = eval_util.EvaluationMetrics(data_reader.num_classes, 20)
	for i in range(3844 // batch_size):
		batch = next(numpy_dataset)
		test_input = tf.convert_to_tensor(batch[0])
		test_labels = tf.convert_to_tensor(batch[1])

		predictions = model.predict(test_input)
		
		loss_vals = loss.eval_loss(test_labels, predictions)

		test_labels = test_labels.numpy()
		loss_vals = loss_vals.numpy()

		evaluation_metrics.accumulate(predictions, test_labels, loss_vals)
	eval_dict = evaluation_metrics.get()

	print(eval_dict)


	#model.save_weights('/home/conorfvedova_google_com/saved_model/model-final.h5')

if __name__ == "__main__":
	train()
