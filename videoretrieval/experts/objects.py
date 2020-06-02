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

Class for managing object detector experts.
"""

from base import BaseExpert

class ResNext101(BaseExpert):
    """Implementation of the ResNext expert class."""

    @property
    def name(self):
        return "resnext"
    
    @property
    def embedding_shape(self):
        return (2048,)

class SeNet154(BaseExpert):
    """Implementation of the SeNet expert class."""

    @property
    def name(self):
        return "senet"
    
    @property
    def embedding_shape(self):
        return (2048,)
