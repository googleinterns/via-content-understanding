"""Implementation of the text encoder.

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
import tensorflow as tf
import metrics.rankings

class EncoderModel(tf.keras.Model):
    """An implementation of an Encoder model."""

    def __init__(self, video_encoder, text_encoder, loss_hyperparameter_m):
        """Intialize an encoder with a video encoder and a text encoder.

        Parameters:
            video_encoder: the Video Encoder to be used.
            text_encoder: the Text Encoder to be used.
            loss_hyperparameter_m: TODO(ryanehrlich).
        """
        super(EncoderModel, self).__init__()

        self.video_encoder = video_encoder
        self.text_encoder = text_encoder
        self.loss_hyperparameter_m = loss_hyperparameter_m

    def compile(
            self, optimizer, loss_fn, recall_at_k_bounds, captions_per_video):
        """Complies the encoder.

        Arguments:
            optimizer: the optimizer for the video encoder.
            loss_fn: the loss function for this model.
        """
        super(EncoderModel, self).compile()

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.recall_at_k_bounds = recall_at_k_bounds

        self.recall_at_k_labels = [f"R_{k}" for k in recall_at_k_bounds]
        self.captions_per_video = captions_per_video

    def train_step(self, video_text_pair_batch):
        """Executes one step of training."""
        video_ids, video_features, text_features, missing_experts = \
            video_text_pair_batch

        with tf.GradientTape() as gradient_tape:
            video_results = self.video_encoder(
                [video_features, missing_experts])
            text_results, mixture_weights = self.text_encoder(text_features)

            loss = self.loss_fn(
                video_results, text_results, mixture_weights, missing_experts,
                self.loss_hyperparameter_m, video_ids)

        gradients = gradient_tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        batch_metrics = {
            label: float("nan") for label in self.recall_at_k_labels}

        batch_metrics["median_rank"] = float("nan")
        batch_metrics["mean_rank"] = float("nan")

        batch_metrics["loss"] = loss        

        return batch_metrics

    def remove_repeated_video_data(self, tensor):
        """Removes repeated video data from a tensor.

        Because the dataset is constructed on a pairwise basis, if there are
        multiple videos in a batch, there will be repeated video data. This
        removes the repeated data.

        Parameters:
            tensor: the tensor with repeated video data

        Returns: a tensor like the `tensor` inputted with repeated video data
        removed.
        """
        return tensor[::self.captions_per_video]

    def test_step(self, video_text_pair_batch):
        """Executes one test step."""
        video_ids, video_features, text_features, missing_experts = \
            video_text_pair_batch

        assert video_ids.shape[0] % self.captions_per_video == 0

        video_ids = self.remove_repeated_video_data(video_ids)
        video_features = map(self.remove_repeated_video_data, video_features)
        missing_experts = self.remove_repeated_video_data(missing_experts)

        video_results = self.video_encoder(
            [video_features, missing_experts])
        text_results, mixture_weights = self.text_encoder(text_features)

        valid_metrics = {}
        loss = []
        ranks = []

        # Because there are multiple captions per video, we shard the embeddings
        # into self.captions_per_video shards. Because the video data is the
        # doesn't change over multiple captions, splitting the data into shards
        # and computing retrieval methods on shards instead of computing metrics
        # on the entire validation set at once is the cleaner option.

        for caption_index in range(self.captions_per_video):
            shard_text_results = [embed[caption_index::self.captions_per_video]
                for embed in text_results]
            shard_mixture_weights = mixture_weights[
                caption_index::self.captions_per_video]

            loss.append(self.loss_fn(
                video_results,
                shard_text_results,
                shard_mixture_weights,
                missing_experts,
                self.loss_hyperparameter_m, video_ids))

            ranks.append(metrics.rankings.compute_ranks(
                shard_text_results, shard_mixture_weights, video_results,
                missing_experts))

        valid_metrics["loss"] = tf.reduce_mean(tf.stack(loss))
        ranks = tf.concat(ranks, axis=0)

        valid_metrics["mean_rank"] = metrics.rankings.get_mean_rank(ranks)
        valid_metrics["median_rank"] = metrics.rankings.get_median_rank(ranks)

        for k, label in zip(self.recall_at_k_bounds, self.recall_at_k_labels):
            valid_metrics[label] = metrics.rankings.get_recall_at_k(ranks, k)

        return valid_metrics
