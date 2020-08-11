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
from abc import ABC as abstract_class, abstract_method

class BERTLikeModel(BaseLanguageModel, abstract_class):
    """An implementation of BaseLanguageModel for BERT like models."""
    max_input_length = 37
    _batch_size = 32
    _contextual_embeddings_shape = 768
    
    @abstactmethod
    def __init__(self):
        pass

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def encoded_shape(self):
        return (self.max_input_length,)

    @property
    def contextual_embeddings_shape(self):
        return (self.max_input_length, self._contextual_embeddings_shape)

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
        return [self.model(ids, attention_mask=attention_mask)[0]]

class BERTLargeModel(BERTLikeModel):
    """An implementation of BaseLanguageModel for the BERT Large."""
    _contextual_embeddings_shape = 1024

    def __init__(self):
        self.model = TFBertModel.from_pretrained("bert-large-uncased")
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")
        self.tokenizer.model_max_length = self.max_input_length

    @property
    def name(self):
        return "bert_large"

class BERTModel(BERTLikeModel):
    """An implementation of BaseLanguageModel for BERT Base."""

    def __init__(self):
        self.model = TFBertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.tokenizer.model_max_length = self.max_input_length

    @property
    def name(self):
        return "bert_base"

class RobertaModel(BERTLikeModel):
    """An implementation of BaseLanguageModel for RoBERTa Large."""
    def __init__(self):
        self.model = TFRobertaModel.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.tokenizer.model_max_length = self.max_input_length

    @property
    def name(self):
        return "roberta_base"
