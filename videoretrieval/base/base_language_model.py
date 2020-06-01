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

"""


from abc import ABC as AbstractClass
from abc import abstractmethod

class BaseLanguageModel(AbstractClass):
    """Base class for language models."""

    def __call__(self, ids):
        """Does a forward pass on the language model."""
        return self.forward(ids)

    @property
    @abstractmethod
    def name(self):
        """A short name for a language model, ie: gpt."""
        pass

    @property
    @abstractmethod
    def batch_size(self):
        """The batch size used when inferencing with this model."""
        pass
    

    @abstractmethod
    def forward(self, ids):
        """A forward pass on the model.

        Parameters:
            ids: a batched tensor of ids returned by the method encode.
        Returns: a tensor of contextual embeddings.
        """
        pass

    @abstractmethod
    def encode(self, text):
        """Encode the given text as ids to be passed into the model.

        Parameters:
            text: a string to encode as ids

        Returns:
            A python list of ids zero padded to the appropriate size. 
        """
        pass

