"""Implementation of a model that wraps a video and text encoder.

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
from abc import ABC as abstract_class
from abc import abstractmethod
import math
import tensorflow as tf
import metrics.rankings
from metrics.loss import build_similarity_matrix

class EncoderBaseModel(tf.keras.Model, abstract_class):
    def __init__(
        self, video_encoder, text_encoder, margin_hyperparameter,
        recall_at_k_bounds, captions_per_video):
        super(EncoderBaseModel, self).__init__()
        self.video_encoder = video_encoder
        self.text_encoder = text_encoder
        self.margin_hyperparameter = margin_hyperparameter

        self.recall_at_k_bounds = recall_at_k_bounds
        self.recall_at_k_labels = [f"R_{k}" for k in recall_at_k_bounds]
        self.captions_per_video = captions_per_video

    def compile(self, optimizer, loss_function):
        """Complies this model.

        Arguments:
            optimizer: the optimizer for the video encoder.
            loss_fn: the loss function for this model.
            recall_at_k_bounds: the a list of integers to use as thresholds when
                computing recall at k.
            captions_per_video: the number of captions associated with each
                video.
        """
        super(EncoderBaseModel, self).compile()

        self.optimizer = optimizer
        self.loss_function = loss_function

    def train_step(self, video_text_pair_batch):
        """Executes one step of training.

        Args:
            video_text_pair_batch: the data to be inputted to the forward_pass
                function.
        """
        missing_experts = video_text_pair_batch[-1]

        with tf.GradientTape() as gradient_tape:
            encoder_output = self.forward_pass(
                video_text_pair_batch, training=True)
            similarity_matrix = build_similarity_matrix(
                *(*encoder_output, missing_experts))
            loss = self.loss_function(
                similarity_matrix, self.margin_hyperparameter)

        gradients = gradient_tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # It's wasteful to calculate ranking metrics for the entire train
        # dataset, so we just mark the values as NaN for keras.
        batch_metrics = {
            label: float("nan") for label in self.recall_at_k_labels}
        batch_metrics["median_rank"] = float("nan")
        batch_metrics["mean_rank"] = float("nan")
        batch_metrics["loss"] = loss        

        return batch_metrics

    def remove_repeated_data(self, tensor):
        """Removes repeated video data from a tensor.

        Because the dataset is constructed on a pairwise basis, if there are
        multiple videos in a batch, there will be repeated video data. This
        removes the repeated data.

        Parameters:
            tensor: the tensor with repeated video data. The data in tensor
                should be repeated self.captions_per_video times.

        Returns: a tensor like the `tensor` inputted with repeated video data
        removed.
        """
        return tensor[::self.captions_per_video]

    def remove_repeated_video_data(self, video_text_pair_batch):
        """Removes repeated video data from a batch."""
        video_text_pair_batch = list(video_text_pair_batch)
        video_ids = video_text_pair_batch[0]
        video_features = video_text_pair_batch[1]
        missing_experts = video_text_pair_batch[-1]

        video_ids = self.remove_repeated_data(video_ids)
        video_features = list(map(
            self.remove_repeated_data, video_features))
        missing_experts = self.remove_repeated_data(missing_experts)

        video_text_pair_batch[0] = video_ids
        video_text_pair_batch[1] = video_features
        video_text_pair_batch[-1] = missing_experts

        return tuple(video_text_pair_batch)

    def test_step(self, video_text_pair_batch):
        """Executes one test step.

        Args:
            video_text_pair_batch: input to the forward_pass function.
                Additionally, for each video caption pair in this tuple, each
                video must have self.num_captions_per_video associated with it.
                Each video caption pair also must be adjacent to all other video
                caption pairs for the same video. 
        """
        video_text_pair_batch = self.remove_repeated_video_data(
            video_text_pair_batch)
        missing_experts = video_text_pair_batch[-1]

        video_results, text_results, mixture_weights = self.forward_pass(
            video_text_pair_batch, training=False)

        valid_metrics = {}
        loss = []
        ranks = []

        # Because there are multiple captions per video, we shard the embeddings
        # into self.captions_per_video shards. Because the video data is
        # repeated multiple times in a given batch, splitting the data and
        # computing retrieval methods on shards instead of computing metrics on 
        # the entire validation set at once is the cleaner option.
        for caption_index in range(self.captions_per_video):
            shard_text_results = [embed[caption_index::self.captions_per_video]
                for embed in text_results]
            shard_mixture_weights = mixture_weights[
                caption_index::self.captions_per_video]

            similarity_matrix = build_similarity_matrix(
                video_results,
                shard_text_results,
                shard_mixture_weights,
                missing_experts)

            loss.append(self.loss_function(
                similarity_matrix, self.margin_hyperparameter))
            ranks.append(metrics.rankings.compute_ranks(similarity_matrix))
        
        ranks = tf.concat(ranks, axis=0)

        valid_metrics["loss"] = tf.reduce_mean(tf.stack(loss))
        valid_metrics["mean_rank"] = metrics.rankings.get_mean_rank(ranks)
        valid_metrics["median_rank"] = metrics.rankings.get_median_rank(ranks)

        for k, label in zip(self.recall_at_k_bounds, self.recall_at_k_labels):
            valid_metrics[label] = metrics.rankings.get_recall_at_k(ranks, k)

        return valid_metrics

    @abstractmethod
    def forward_pass(self, data, training=False):
        """Executes a forward pass with the given data."""

class EncoderForFrozenLanguageModel(EncoderBaseModel):
    """An implementation of a keras model that trains an arbitrary Text Encoder
    in concert with an arbitrary Video Encoder.

    Attributes:
        video_encoder: The encoder used to encode features from videos.
        text_encoder: The encoder used to encode features from text.
    """

    def forward_pass(self, input_data, training=False):
        _, video_data, text_data, missing_experts = input_data

        video_embeddings = self.video_encoder([video_data, missing_experts])
        text_embeddings, mixture_weights = self.text_encoder(text_data)

        return video_embeddings, text_embeddings, mixture_weights

class EncoderForLanguageModelTuning(EncoderBaseModel):
    """An implementation of an Encoder model.

    This model wraps a video and text encoder to help with training them.

    Attributes:
        video_encoder: The encoder used to encode features from videos.
        text_encoder: The encoder used to encode features from text.
    """
    def __init__(
        self, video_encoder, text_encoder, margin_hyperparameter,
        recall_at_k_bounds, captions_per_video, language_model,
        language_model_batch_size):
        super(EncoderForLanguageModelTuning, self).__init__(
            video_encoder, text_encoder, margin_hyperparameter,
            recall_at_k_bounds, captions_per_video)

        self.language_model = language_model
        self.language_model_batch_size = language_model_batch_size

    def language_model_forward_pass(
        self, text_tokens, attention_mask, training=False):
        """Executes a forward pass on the language model."""
        embeddings = []
        num_tokens = text_tokens.shape[0]

        batches = math.ceil(num_tokens / self.language_model_batch_size)

        for index in range(batches):
            start_index = self.language_model_batch_size * index
            end_index = start_index + self.language_model_batch_size
            text_tokens_shard = text_tokens[start_index:end_index]
            attention_mask_shard = attention_mask[start_index:end_index]
            
            language_model_output = self.language_model(
                text_tokens_shard,
                attention_mask=attention_mask_shard,
                training=training)[0]

            # Zero out missing embeddings
            zero_masked_output = language_model_output * tf.cast(
                attention_mask_shard[:, :, None], tf.float32)
            embeddings.append(zero_masked_output)

        return tf.concat(embeddings, axis=0)

    def forward_pass(self, input_data, training=False):
        video_features = input_data[1]
        text_tokens = input_data[2]
        attention_masks = input_data[3]
        missing_experts = input_data[4]

        video_embeddings = self.video_encoder([video_features, missing_experts])
        contextual_embeddings = self.language_model_forward_pass(
            text_tokens, attention_masks, training)
        text_embeddings, mixture_weights = self.text_encoder(
            contextual_embeddings)

        return video_embeddings, text_embeddings, mixture_weights
