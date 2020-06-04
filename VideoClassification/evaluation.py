import tensorflow as tf
import tensorflow_datasets as tfds

import eval_util
import NetVLAD_CG
import reader_utils
import readers
import loss

def test(model_dir, num_clusters=64, batch_size=64, iterations=None, random_frames=True, num_mixtures=2, fc_units=2048):
	data_reader = reader_utils.get_reader()

	test_dataset = data_reader.get_dataset('/home/conorfvedova_google_com/data/test/', batch_size=batch_size, num_workers=8, type="test")

	numpy_dataset = tfds.as_numpy(test_dataset)

	num_frames = data_reader.max_frames

	video_input_shape = (batch_size, num_frames, 1024)
	audio_input_shape = (batch_size, num_frames, 128)

	#Compile and train model
	model = NetVLAD_CG.VideoClassifier(num_clusters, video_input_shape, audio_input_shape, fc_units=fc_units, iterations=iterations, random_frames=random_frames, num_classes=data_reader.num_classes, num_mixtures=num_mixtures)
	print(model.layers)
	model.load_weights(model_dir, by_name=True)
	model.compile()

	evaluation_metrics = eval_util.EvaluationMetrics(data_reader.num_classes, 20)
	for batch in numpy_dataset:
		test_input = tf.convert_to_tensor(batch[0])
		test_labels = tf.convert_to_tensor(batch[1])

		predictions = model.predict(test_input)
		
		loss_vals = loss.eval_loss(test_labels, predictions)
		print(loss_vals.shape)
		print(predictions.shape)
		print(test_labels.shape)

		predictions = predictions.numpy()
		test_labels = test_labels.numpy()
		loss_vals = loss_vals.numpy()

		evaluation_metrics.accumulate(predictions, test_labels, loss_vals)
	eval_dict = evaluation_metrics.get()

	print(eval_dict)

if __name__ == "__main__":
	test("/home/conorfvedova_google_com/saved_model/model-final.h5")
