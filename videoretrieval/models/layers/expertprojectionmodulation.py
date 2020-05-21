"""Implementation of the Expert Projection Modulation Layer.

Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import tensorflow as tf

class ExpertProjectionModulationLayer(tf.keras.layers.Layer):
    """An implementation of the expert projection modulation layer.

	The expert projection modulation takes two inputs: first, the temporally
	aggregated expert feature vector, and second, the attention vector for that
	feature. The output of this layer is the feature vector multipled
	element-wise with the sigmoid activiations of the attention vector.
    """
    def __init__(self):
        super(ExpertProjectionModulationLayer, self).__init__()

    def call(self, inputs):
    	"""Forward pass on the expert projection modulation layer.

		Parameters:
			inputs: an array of exactly two elements. The first element should
				be the feature vector, and the second should be the attention
				vector.
    	"""
        assert len(inputs) == 2

        feature_vector, attention_vector = inputs

        return feature_vector * tf.math.sigmoid(attention_vector)
