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
    """An implementation of BaseLanguageModel for BERT Large."""
    @property
    def name(self):
        return "bert_large"

    @property
    def batch_size(self):
        return 32

    @property
    def max_input_length(self):
        return 37

    def get_tokenizer(self):
        return BertTokenizerFast.from_pretrained("bert-large-uncased")

    def get_model(self):
        return TFBertModel.from_pretrained("bert-large-uncased")

    @property
    def contextual_embeddings_shape(self):
        return (37, 1024)

    def forward(self, ids, attention_mask):
        return [super(BERTLargeModel, self).forward(ids, attention_mask)[0]]

class BERTModel(BaseLanguageModel):
    """An implementation of BaseLanguageModel for BERT Base."""
    @property
    def name(self):
        return "bert_base"

    @property
    def batch_size(self):
        return 32

    @property
    def max_input_length(self):
        return 37

    def get_tokenizer(self):
        return BertTokenizerFast.from_pretrained("bert-base-uncased")

    def get_model(self):
        return TFBertModel.from_pretrained("bert-base-uncased")

    @property
    def contextual_embeddings_shape(self):
        return (37, 768)

    def forward(self, ids, attention_mask):
        return [super(BERTModel, self).forward(ids, attention_mask)[0]]

class RobertaModel(BaseLanguageModel):
    """An implementation of BaseLanguageModel for RoBERTa Base."""
    @property
    def name(self):
        return "roberta_base"

    @property
    def batch_size(self):
        return 32

    @property
    def max_input_length(self):
        return 37

    def get_tokenizer(self):
        return RobertaTokenizerFast.from_pretrained("roberta-base")

    def get_model(self):
        return TFRobertaModel.from_pretrained("roberta-base")

    @property
    def contextual_embeddings_shape(self):
        return (37, 768)

    def forward(self, ids, attention_mask):
        return [super(RobertaModel, self).forward(ids, attention_mask)[0]]
