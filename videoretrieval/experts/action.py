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

Class for managing action recongition experts.
"""

from base import BaseExpert

class I3D(BaseExpert):
    """Implementation of the I3D expert class."""

    @property
    def name(self):
        return "i3d"
    
    @property
    def embedding_shape(self):
        return (1024,)

class R2P1D(BaseExpert):
    """Implementation of the R(2 + 1)d expert class."""

    @property
    def name(self):
        return "r2p1d"
    
    @property
    def embedding_shape(self):
        return (512,)
