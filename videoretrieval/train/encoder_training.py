
import tensorflow as tf
from metrics.loss import bidirectional_max_margin_ranking_loss 

def get_train_step(video_encoder, text_encoder):
    video_encoder_optimizer = tf.keras.optimizers.Adam()
    text_encoder_optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')


    @tf.function
    def train_step(video_embeddings_batch, text_embeddings_batch):
        with tf.GradientTape() as video_tape, tf.GradientTape() as text_tape:
            video_embeddings = video_encoder(video_embeddings_batch)
            text_embeddings = text_encoder(text_embeddings_batch)

        loss = bidirectional_max_margin_ranking_loss(
            video_embeddings,
            text_embeddings)

        video_gradients = video_tape.gradient(
            loss, video_encoder.trainable_variables)
        text_gradients = text_tape.gradient(
            loss, text_encoder.trainable_variables)

        video_encoder_optimizer.apply_gradients(zip(
            video_gradients, video_embeddings.trainable_variables))
        text_encoder_optimizer.apply_gradients(zip(
            text_gradients, text_embeddings.trainable_variables))

        train_loss(loss)

    return train_step, train_loss, valid_loss 

def epoch(dataset, train_step_function, batch_size):
    batched_dataset = dataset.batch(batch_size).prefetch()

    for video_embeddings_batch, text_embeddings_batch in batched_dataset:
        train_step_function(video_embeddings_batch, text_embeddings_batch)



def epochs(num_of_epochs):
    pass