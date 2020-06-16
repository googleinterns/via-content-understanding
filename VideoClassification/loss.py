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

Defines the loss function for the model.
"""

import tensorflow as tf

def custom_crossentropy(y_actual, y_predicted, epsilon=10e-6, alpha=0.5):
	float_labels = tf.cast(y_actual, tf.float32)

	cross_entropy_loss = 2*(alpha*float_labels * tf.math.log(y_predicted + epsilon) + (1-alpha)*(1 - float_labels) * tf.math.log(1 - y_predicted + epsilon))

	cross_entropy_loss = tf.math.negative(cross_entropy_loss)
	return tf.math.reduce_mean(tf.math.reduce_sum(cross_entropy_loss, 1))
