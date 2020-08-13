"""Implementation of BaseLanguageModel class for gpt-1.

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

from base import BaseLanguageModel
from transformers import TFOpenAIGPTModel, OpenAIGPTTokenizerFast

class OpenAIGPTModel(BaseLanguageModel):
    """An implementation of BaseLanguageModel for the openai-gpt1 model."""

    @property
    def name(self):
        return "openai_gpt"

    @property
    def batch_size(self):
        return 42

    @property
    def max_input_length(self):
        return 37

    def get_tokenizer(self):
        tokenizer = OpenAIGPTTokenizerFast.from_pretrained("openai-gpt")
        tokenizer.pad_token = " "
        return tokenizer

    def get_model(self):
        return TFOpenAIGPTModel.from_pretrained("openai-gpt")

    @property
    def contextual_embeddings_shape(self):
        return (37, 768)
