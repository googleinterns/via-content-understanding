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

Functions for training encoders.
"""
import tensorflow as tf
from metrics.loss import bidirectional_max_margin_ranking_loss 
from tqdm import tqdm as terminal_progress_bar
from tqdm import tqdm_notebook as progress_bar_notebook
import math

def get_train_step(video_encoder, text_encoder, m):
    video_encoder_optimizer = tf.keras.optimizers.Adam()
    text_encoder_optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')

    @tf.function
    def forward(video_embeddings_batch, text_embeddings_batch):
        video_embeddings = video_encoder(video_embeddings_batch)
        text_results = text_encoder(text_embeddings_batch)

        return video_embeddings, text_results

    @tf.function
    def train_step(video_embeddings_batch, text_embeddings_batch):
        with tf.GradientTape() as video_tape, tf.GradientTape() as text_tape:
            video_results = video_encoder(video_embeddings_batch)
            text_results = text_encoder(text_embeddings_batch)

            loss = bidirectional_max_margin_ranking_loss(
                video_results, text_results, m)

        video_gradients = video_tape.gradient(
            loss, video_encoder.trainable_variables)
        text_gradients = text_tape.gradient(
            loss, text_encoder.trainable_variables)

        video_encoder_optimizer.apply_gradients(zip(
            video_gradients, video_encoder.trainable_variables))
        text_encoder_optimizer.apply_gradients(zip(
            text_gradients, text_encoder.trainable_variables))

        train_loss(loss)

    return train_step, forward, train_loss, valid_loss 

def epoch(train_ds, valid_ds, train_ds_len, valid_ds_len, train_step_function,
    forward_function, batch_size, train_loss, valid_loss, m, in_notebook=False,
    batches_per_buffer=50):
    train_loss.reset_states()
    valid_loss.reset_states()

    train_batched_dataset = (train_ds
        .shuffle(batches_per_buffer*batch_size)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE))

    valid_batched_dataset = (valid_ds
        .shuffle(batches_per_buffer*batch_size)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE))

    if in_notebook:
        progress_bar = progress_bar_notebook
    else:
        progress_bar = terminal_progress_bar

    train_iter_progress = progress_bar(
        iter(train_batched_dataset), total=math.ceil(train_ds_len / batch_size))

    for video_embeddings_batch, text_embeddings_batch in train_iter_progress:
        train_step_function(video_embeddings_batch, text_embeddings_batch)
        train_iter_progress.set_description(
            f"Train Loss: {train_loss.result().numpy()}")

    valid_iter_progress = progress_bar(
        iter(valid_batched_dataset), total=math.ceil(valid_ds_len / batch_size))

    for video_embeddings_batch, text_embeddings_batch in valid_iter_progress:

        video_embeddings, text_embeddings = forward_function(
                video_embeddings_batch, text_embeddings_batch)

        valid_loss(bidirectional_max_margin_ranking_loss(
            video_embeddings,text_embeddings, m))
        valid_iter_progress.set_description(
            f"Valid Loss: {valid_loss.result().numpy()}")

    print()

def fit(
    video_encoder,
    text_encoder,
    num_epochs,
    train_ds,
    valid_ds,
    train_ds_len,
    valid_ds_len,
    batch_size,
    m,
    in_notebook=False):
    
    train_step, forward, train_loss, valid_loss = \
        get_train_step(video_encoder, text_encoder, m)

    for epoch_num in range(num_epochs):
        print(f"Epoch {epoch_num+1} / {num_epochs}")
        epoch(
            train_ds,
            valid_ds,
            train_ds_len,
            valid_ds_len, 
            train_step,
            forward,
            batch_size,
            train_loss,
            valid_loss,
            m,
            in_notebook)
