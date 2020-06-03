import tensorflow as tf

def custom_crossentropy(y_actual, y_predicted, epsilon=10e-6, alpha=0.5):
	float_labels = tf.cast(y_actual, tf.float32)

	cross_entropy_loss = 2*(alpha*float_labels * tf.math.log(y_predicted + epsilon) + (1-alpha)*(1 - float_labels) * tf.math.log(1 - y_predicted + epsilon))

	cross_entropy_loss = tf.math.negative(cross_entropy_loss)
	return tf.math.reduce_mean(tf.math.reduce_sum(cross_entropy_loss, 1))