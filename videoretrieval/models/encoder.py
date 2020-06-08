import tensorflow as tf

class EncoderModel(tf.keras.Model):
    def __init__(self, video_encoder, text_encoder, loss_hyperparameter_m):
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
        video_features, text_features = video_text_pair_batch

        with tf.GradientTape() as video_tape, tf.GradientTape() as text_tape:
            video_results = self.video_encoder(video_features)
            text_results = self.text_encoder(text_features)

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
        video_features, text_features = video_text_pair_batch
        
        video_results = self.video_encoder(video_features)
        text_results = self.text_encoder(text_features)

        loss = self.loss_fn(
            video_results, text_results, self.loss_hyperparameter_m)

        return {"loss": loss}
