from base import BaseLanguageModel
from transformers import BertTokenizerFast, TFBertModel
class BERTModel(BaseLanguageModel):
    """An implementation of BaseLanguageModel for the openai-gpt1 model."""

    max_input_length = 37
    _batch_size = 42

    def __init__(self):
        self.model = TFBertModel.from_pretrained("bert-large-uncased")
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")
        self.tokenizer.model_max_length = _batch_size

    @property
    def name(self):
        return "bert_large"

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def encoded_shape(self):
        return (self.max_input_length,)

    @property
    def contextual_embeddings_shape(self):
        return (37, 1024)
    

    def encode(self, text):
        """Encode the given text as ids to be passed into the model.

        Parameters:
            text: a string to encode.

        Returns:
            A tuple of two elements. First, a python list of ids zero padded to
            the appropriate size. Second, the number of tokens the text was
            encoded to.
        """

        tokenized = self.tokenizer(text),

        input_ids = tokenized["input_ids"]

        return input_ids, input_ids.index(0)

    def forward(self, ids):
        """A forward pass on the model.

        Parameters:
            ids: a batched tensor of ids returned by the method encode.

        Returns: a tensor of contextual embeddings.
        """

        return self.model(ids)
