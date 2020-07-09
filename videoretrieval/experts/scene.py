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

Class for managing scene experts.
"""

from base import BaseExpert

class DenseNet161(BaseExpert):
    """Implementation of the DenseNet expert class."""

    @property
    def name(self):
        return "densenet"
    
    @property
    def embedding_shape(self):
        return (2208,)

    def feature_transformation(self, feature):
        """Removes an unnecessary dimension from feature.

        In the cache, the densenet features are stored in a vector with a shape
        of 1 x 2208. This transformation transforms the vector into a vector
        with shape 2208.
        """ 
    	return feature[0]