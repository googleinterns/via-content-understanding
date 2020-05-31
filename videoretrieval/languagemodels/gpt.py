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

Functions to download, extract, and process precomputed features.
"""

from base import BaseLanguageModel
from transformers import TFOpenAIGPTModel, OpenAIGPTTokenizer

class OpenAIGPTModel(BaseLanguageModel):

    max_input_length = 324
    _batch_size = 48

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
        return (324,)

    def pad_tokens(self, tokens):
        if len(tokens) >= self.max_input_length:
            return tokens[:self.max_input_length]
        else:
            return tokens + [0] * (self.max_input_length - len(tokens))

    def encode(self, text):
        tokens = self.tokenizer.tokenize(text)
        padded_tokens = self.pad_tokens(tokens)

        return self.tokenizer.convert_tokens_to_ids(padded_tokens)

    def forward(self, ids):
        return self.model.test_on_batch(ids)
