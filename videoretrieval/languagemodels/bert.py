"""Implementation of BaseLanguageModel class for BERT models.

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
from transformers import BertTokenizerFast, TFBertModel
from transformers import RobertaTokenizerFast, TFRobertaModel

class BERTLargeModel(BaseLanguageModel):
    """An implementation of BaseLanguageModel for the BERT Large."""
    @property
    def name(self):
        return "bert_large"

    @property
    def batch_size(self):
        return 32

    def get_max_input_length(self):
        return 37

    @abstractmethod
    def get_tokenizer(self):
        return BertTokenizerFast.from_pretrained("bert-large-uncased")

    @abstractmethod
    def get_model(self):
        return TFBertModel.from_pretrained("bert-large-uncased")

    @property
    @abstractmethod
    def contextual_embeddings_shape(self):
        return (37, 1024)

class BERTModel(BERTLikeModel):
    """An implementation of BaseLanguageModel for BERT Base."""
    @property
    def name(self):
        return "bert_large"

    @property
    def batch_size(self):
        return 32

    def get_max_input_length(self):
        return 37

    @abstractmethod
    def get_tokenizer(self):
        return BertTokenizerFast.from_pretrained("bert-base-uncased")

    @abstractmethod
    def get_model(self):
        return TFBertModel.from_pretrained("bert-base-uncased")

    @property
    @abstractmethod
    def contextual_embeddings_shape(self):
        return (37, 768)

class RobertaModel(BERTLikeModel):
    """An implementation of BaseLanguageModel for RoBERTa Base."""
    def __init__(self):
        self.model = TFRobertaModel.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.tokenizer.model_max_length = self.max_input_length

    @property
    def name(self):
        return "roberta_base"

    @property
    def batch_size(self):
        return 32

    def get_max_input_length(self):
        return 37

    @abstractmethod
    def get_tokenizer(self):
        return RobertaTokenizerFast.from_pretrained("roberta-base")

    @abstractmethod
    def get_model(self):
        return TFRobertaModel.from_pretrained("roberta-base")

    @property
    @abstractmethod
    def contextual_embeddings_shape(self):
        return (37, 768)
