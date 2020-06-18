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

    def compile(self, video_encoder_optimizer, text_encoder_optimizer, loss_fn,
            text_data_shape=None):
        super(EncoderModel, self).compile()

        self.video_encoder_optimizer = video_encoder_optimizer
        self.text_encoder_optimizer = text_encoder_optimizer
        self.loss_fn = loss_fn

    def train_step(self, video_text_pair_batch):
        video_features, text_features, missing_experts = video_text_pair_batch

        with tf.GradientTape() as video_tape, tf.GradientTape() as text_tape:
            video_results = self.video_encoder(video_features)
            text_results = self.text_encoder(text_features)

            text_results = self.zero_missing_modalities(
                text_results, missing_experts)

            loss = self.loss_fn(
                video_results, text_results, self.loss_hyperparameter_m)

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
        video_features, text_features, missing_experts = video_text_pair_batch
        
        video_results = self.video_encoder(video_features)
        text_results = self.text_encoder(text_features)

        text_results = self.zero_missing_modalities(
            text_results, missing_experts)

        loss = self.loss_fn(
            video_results, text_results, self.loss_hyperparameter_m)

        return {"loss": loss}

    def build_missing_modalities_mask(self, missing_experts):
        num_experts = self.text_encoder.num_of_experts

        return tf.repeat(
            missing_experts,
            [self.text_encoder.encoded_expert_dimensionality] * num_experts, 
            axis=-1)

    def zero_missing_modalities(
        self, text_results, missing_experts):
        zero_mask = self.build_missing_modalities_mask(missing_experts)

        return tf.math.l2_normalize(text_results * zero_mask, axis=-1)
