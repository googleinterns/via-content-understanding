
import tensorflow as tf
from metrics.loss import bidirectional_max_margin_ranking_loss 
from tqdm import tqdm as terminal_progress_bar
from tqdm import tqdm_notebook as progress_bar_notebook

def get_train_step(video_encoder, text_encoder):
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
                video_results, text_results, 1)

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
    forward_function, batch_size, train_loss, valid_loss, in_notebook=False):
    train_loss.reset_states()
    valid_loss.reset_states()

    train_batched_dataset = (train_ds
        .shuffle(50*batch_size)
        .batch(batch_size))

    valid_batched_dataset = (valid_ds
        .shuffle(50*batch_size)
        .batch(batch_size))

    if in_notebook:
        progress_bar = progress_bar_notebook
    else:
        progress_bar = terminal_progress_bar

    train_iter_progress = progress_bar(
        iter(train_batched_dataset), total=train_ds_len)

    for video_embeddings_batch, text_embeddings_batch in train_iter_progress:
        train_step_function(video_embeddings_batch, text_embeddings_batch)
        train_iter_progress.set_description(
            f"Train Loss: {train_loss.result().numpy()}")

    valid_iter_progress = progress_bar(
        iter(valid_batched_dataset), total=valid_ds_len)

    for video_embeddings_batch, text_embeddings_batch in valid_iter_progress:
        valid_loss(bidirectional_max_margin_ranking_loss(
            *forward_function(
                video_embeddings_batch, text_embeddings_batch, 1)))
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
    batch_size):
    
    train_step, forward, train_loss, valid_loss = \
        get_train_step(video_encoder, text_encoder)

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
            valid_loss)
