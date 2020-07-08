"""Implementation of the temporal aggregation layer.

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
from .netvlad import NetVLAD

class TemporalAggregationLayer(tf.keras.layers.Layer):
    """A layer that aggregates expert features to a common dimensionality.

    If the expert feature is variable length, netvlad is used project the
    feature to a fixed length vector. Then, the fixed length vector is projected
    to a specified dimensionality and then l2 normalized. 

    If the expert feature is constant length, the feature is projected to a
    specified dimensionality and then l2 normalized.

    Attributes:
        output_dim: the dimensionality the experts are aggregated to.
        use_netvald: a boolean indicating if netvlad is used for aggregation.
        netvlad_clusters: the number of clusters the netvlad layer uses. 
        netvlad: a netvlad layer used to aggregate features to a fixed length.
            This will be missing if the use_netvlad attribute is False.
        projection_layer: a dense layer used to project the expert features to a
            common dimensionality.
     """

    def __init__(
        self, output_dim, use_netvlad, kernel_initializer, bias_initializer, 
        netvlad_clusters=5, ghost_clusters=0):
        """Initializers this temporal aggregation layer.

        Arguments:
            output_dim: the dimensionality to project the experts to.
            use_netvlad: if NetVLAD should be used to aggregate features.
            kernel_initializer: the way to initialize the dense layer's kernels.
            bias_initializer: the way to initialize the dense layer's biases.
            netvlad_clusers: the number of clusters NetVLAD should have.
            ghost_clusters: the number of ghost clusters NetVLAD should have.    
        """
        super(TemporalAggregationLayer, self).__init__()

        self.output_dim = output_dim
        self.use_netvlad = use_netvlad
        self.netvlad_clusters = netvlad_clusters

        if self.use_netvlad:
            self.netvlad = NetVLAD(self.netvlad_clusters, ghost_clusters)

        self.projection_layer = tf.keras.layers.Dense(self.output_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)

    def call(self, input_):
        if self.use_netvlad:
            features = self.netvlad(input_)
        else:
            features = input_

        output = self.projection_layer(features)
        return tf.math.l2_normalize(output, axis=-1)
