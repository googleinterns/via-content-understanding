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

    def __init__(self):
        self.model = TFOpenAIGPTModel.from_pretrained("openai-gpt")
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

    def encode(self, text):
        self.tokenizer.encode(text)

    def forward(self, ids):
        return self.model([ids])
