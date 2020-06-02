""" Copyright 2020 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Defines a base class for experts.
"""

from abc import ABC as AbstractClass
from abc import abstractmethod
import pathlib

class BaseExpert(AbstractClass):
    """Base class for expert models."""

    @property
    @abstractmethod
    def name(self):
        """A short name for the expert: e.g., slowfast."""
        pass

    @property
    @abstractmethod
    def embedding_shape(self):
        """The shape of the embedding outputted by the expert."""
        pass
