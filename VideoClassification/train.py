import tensorflow as tf
import NetVLAD_CG
import reader_utils

import readers

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
NUM_EXAMPLES = 3844

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "~/data/train/train*.tfrecord",
                      "The directory to save the model files in.")
  flags.DEFINE_string(
      "train_data_pattern", "~/data/train/train*.tfrecord",
      "File glob for the training dataset. If the files refer to Frame Level "
      "features (i.e. tensorflow.SequenceExample), then set --reader_type "
      "format. The (Sequence)Examples are expected to have 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")


def train(epochs=100, lr=0.01, num_clusters=100, batch_size=1024, iterations=None, random_frames=True, num_mixtures=2):
	steps_per_epoch = NUM_EXAMPLES / epochs
        
        #Set up Reader and Preprocess Data
	reader = reader_utils.get_reader()

	unused_video_id, model_input_raw, labels_batch, num_frames = (
      reader_utils.get_input_data_tensors(
          reader,
          '~/data/train/train*.tfrecord',
          batch_size=batch_size,
          num_readers=8,
          num_epochs=epochs))
	  
	feature_dim = len(model_input_raw.get_shape()) - 1

	model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
	video_input_shape = (batch_size, num_frames, 1024)
	audio_input_shape = (batch_size, num_frames, 128)

	#Compile and train model
	model = VideoClassifier(num_clusters, video_input_shape, audio_input_shape, iterations=iterations, random_frames=random_frames, num_classes=reader.num_classes, num_mixtures=num_mixtures)
	
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])

	model.fit(model_input, labels_batch, epochs=epochs)


if __name__ == "__main__":
	train()
