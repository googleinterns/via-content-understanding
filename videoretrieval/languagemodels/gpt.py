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

Implementation of BaseLanguageModel class for gpt-1.
"""

from base import BaseLanguageModel
from transformers import TFOpenAIGPTModel, OpenAIGPTTokenizer

class OpenAIGPTModel(BaseLanguageModel):
    """An implementation of BaseLanguageModel for the openai-gpt1 model."""

    max_input_length = 37
    _batch_size = 42

    def __init__(self):
        self.model = TFOpenAIGPTModel.from_pretrained("openai-gpt")
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

    @property
    def name(self):
        return "openai_gpt"

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def encoded_shape(self):
        return (self.max_input_length,)

    @property
    def contextual_embeddings_shape(self):
        return (37, 768)
    

    def pad_tokens(self, tokens):
        """Zero pad tokens to max input length.

        Parameters:
            tokens: tokens generated by the tokenizer.

        Returns: zero padded tokens to max input length.
        """
        if len(tokens) >= self.max_input_length:
            return tokens[:self.max_input_length]
        else:
            return tokens + [0] * (self.max_input_length - len(tokens))

    def encode(self, text):
        """Encode the given text as ids to be passed into the model.

        Parameters:
            text: a string to encode as ids

        Returns:
            A python list of ids zero padded to the appropriate size. 
        """

        tokens = self.tokenizer.tokenize(text)
        padded_tokens = self.pad_tokens(tokens)

        return self.tokenizer.convert_tokens_to_ids(padded_tokens)

    def forward(self, ids):
        """A forward pass on the model.

        Parameters:
            ids: a batched tensor of ids returned by the method encode.

        Returns: a tensor of contextual embeddings.
        """

        return self.model(ids)
