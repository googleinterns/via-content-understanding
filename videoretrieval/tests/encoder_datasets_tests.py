
import unittest
from train import encoder_datasets
import tensorflow as tf


class TestMakeInferenceDataset(unittest.TestCase):
    """Tests the make inference function in the encoder_datasets module."""
    def test_multiple_videos(self):
        expected_video_data = [
            (0, [0.0]),
            (1, [1.0]),
            (2, [2.0]),
        ]

        expected_text_data = [
            (0, [1.0]),
            (0, [2.0]),
            (0, [3.0]),
            (1, [4.0]),
            (1, [5.0]),
            (2, [6.0]),
            (2, [7.0])
        ]

        video_text_pair_dataset = tf.data.Dataset.from_generator(lambda: [
            ("video0", [0.0], [1.0]),
            ("video0", [0.0], [2.0]),
            ("video0", [0.0], [3.0]),
            ("video1", [1.0], [4.0]),
            ("video1", [1.0], [5.0]),
            ("video2", [2.0], [6.0]),
            ("video2", [2.0], [7.0])], (tf.string, tf.float32, tf.float32))

        video_dataset, text_dataset = encoder_datasets.make_inference_dataset(
            video_text_pair_dataset)

        video_dataset_as_list = list(video_dataset.as_numpy_iterator())
        text_dataset_as_list = list(text_dataset.as_numpy_iterator())

        for generated_ds, expected_ds in [
            (video_dataset_as_list, expected_video_data),
            (text_dataset_as_list, expected_text_data)]:

            for generated, expected in zip(
                generated_ds, expected_ds):

                self.assertEqual(generated[0], expected[0])
                self.assertEqual(list(generated[1]), expected[1])