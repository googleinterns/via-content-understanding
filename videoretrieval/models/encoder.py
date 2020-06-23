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

    def compile(self, video_encoder_optimizer, text_encoder_optimizer, loss_fn):
        """Complies the encoder.

        Arguments:
            video_encoder_optimizer: the optimizer for the video encoder.
            text_encoder_optimizer: the optimizer for the text encoder.
            loss_fn: the loss function for this model.
        """
        super(EncoderModel, self).compile()

        self.video_encoder_optimizer = video_encoder_optimizer
        self.text_encoder_optimizer = text_encoder_optimizer
        self.loss_fn = loss_fn

    def train_step(self, video_text_pair_batch):
        """Executes one step of training."""
        video_ids, video_features, text_features, missing_experts = \
            video_text_pair_batch

        with tf.GradientTape() as video_tape, tf.GradientTape() as text_tape:
            video_results = self.video_encoder(
                [video_features, missing_experts])
            text_results, mixture_weights = self.text_encoder(text_features)

            loss = self.loss_fn(
                video_results, text_results, mixture_weights, missing_experts,
                self.loss_hyperparameter_m, video_ids)

        video_gradients = video_tape.gradient(
            loss, self.video_encoder.trainable_variables)
        text_gradients = text_tape.gradient(
            loss, self.text_encoder.trainable_variables)

        self.video_encoder_optimizer.apply_gradients(zip(
            video_gradients, self.video_encoder.trainable_variables))
        self.text_encoder_optimizer.apply_gradients(zip(
            text_gradients, self.text_encoder.trainable_variables))

        return {"loss": loss}

    def test_step(self, video_text_pair_batch):
        """Executes one test step."""
        video_ids, video_features, text_features, missing_experts = \
            video_text_pair_batch
        
        video_results = self.video_encoder([video_features, missing_experts])
        text_results, mixture_weights = self.text_encoder(text_features)

        loss = self.loss_fn(
            video_results, text_results, mixture_weights, missing_experts,
            self.loss_hyperparameter_m, video_ids)

        return {"loss": loss}
