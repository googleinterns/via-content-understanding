import tensorflow as tf
import NetVLAD_CG
import reader_utils

import readers

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
NUM_EXAMPLES = 3844
NUM_VAL_EXAMPLES = 3844

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


def train(epochs=2, lr=0.01, num_clusters=64, batch_size=64, iterations=None, random_frames=True, num_mixtures=2, fc_units=2048):
	steps_per_epoch = NUM_EXAMPLES // batch_size
	validation_steps = NUM_VAL_EXAMPLES // batch_size
				
	#Set up Reader and Preprocess Data
	reader = reader_utils.get_reader()

	train_dataset = reader.get_dataset('/home/conorfvedova_google_com/data/train/', batch_size=batch_size, num_workers=8)
	print(train_dataset)
	num_frames = reader.max_frames

	validation_dataset = reader.get_dataset('/home/conorfvedova_google_com/data/validate/', batch_size=batch_size, num_workers=8, type="validate")

	video_input_shape = (batch_size, num_frames, 1024)
	audio_input_shape = (batch_size, num_frames, 128)

	#Compile and train model
	model = NetVLAD_CG.VideoClassifier(num_clusters, video_input_shape, audio_input_shape, fc_units=fc_units, iterations=iterations, random_frames=random_frames, num_classes=reader.num_classes, num_mixtures=num_mixtures)
	
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['categorical_accuracy'])

	model.fit(x=train_dataset, steps_per_epoch=steps_per_epoch, validation_data=validation_dataset, validation_steps=validation_steps, epochs=epochs)

	model.save_weights('/home/conorfvedova_google_com/saved_model/model-final.h5')

if __name__ == "__main__":
	train()
