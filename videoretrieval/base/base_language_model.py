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

Defines a base class for language models.
"""


from abc import ABC as AbstractClass
from abc import abstractmethod

class BaseLanguageModel(AbstractClass):
    """Base class for language models."""

    def __init__(self):
        self.model = self.get_model()
        self.tokenizer = self.get_tokenizer()
        self.tokenizer.model_max_length = self.get_max_input_length()

    @property
    @abstractmethod
    def name(self):
        """A short name for a language model, ie: gpt."""

    @property
    @abstractmethod
    def batch_size(self):
        """The batch size used when inferencing with this model."""

    @abstractmethod
    def get_max_input_length(self):
        """The max input length for this model."""

    @abstractmethod
    def get_tokenizer(self):
        """The tokenizer for this model."""

    @abstractmethod
    def get_model(self):
        """The instance of this model."""

    @property
    @abstractmethod
    def contextual_embeddings_shape(self):
        """The output shape of the contextual embeddings.""" 

    @property
    def encoded_shape(self):
        return (self.max_input_length,)

    def encode(self, text):
        """Encode the given text as ids to be passed into the model.

        Parameters:
            text: a string to encode.

        Returns:
            A tuple of two elements. First, a python list of ids zero padded to
            the appropriate size. Second, the attention mask for the ids.
        """
        tokenized = self.tokenizer(text, padding="max_length", truncation=True)
        input_ids = tokenized["input_ids"]
        return input_ids, tokenized["attention_mask"]

    def forward(self, ids, attention_mask):
        """A forward pass on the model.

        Parameters:
            ids: a batched tensor of ids.
            attention_mask: the attention mask for the ids.

        Returns: a tensor of contextual embeddings.
        """
        return self.model(ids, attention_mask=attention_mask)

    def __call__(self, ids, attention_mask):
        """Does a forward pass on the language model."""
        return self.forward(ids, attention_mask)
