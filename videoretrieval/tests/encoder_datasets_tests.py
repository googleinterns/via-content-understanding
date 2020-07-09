
import unittest
from train import encoder_datasets
import tensorflow as tf

def make_mock_video_ids(num_videos):
    video_ids = []

    for video_num in range(num_videos):
        video_ids.append(f"video{video_num}")

def make_mock_id_embeddings_pair_dataset(
    video_ids, mock_tensor_shape, mock_embeddings_per_video=1):
    mock_dataset = []

    for video_id in video_ids:
        for _ in range(mock_embeddings_per_video):
            mock_dataset.append((video_id, tf.zeros(mock_tensor_shape)))

    dataset_generator = (item for item in mock_dataset) 

    return tf.data.Dataset.from_generator(dataset_generator, 
        (tf.string, tf.float32))

def make_mock_precomputed_features(self):


class TestEncoderDatasetsFunctions(unittest.TestCase):

    def test_replacing_video_id_with_expert_features(self):
        mock_video_ids = make_mock_video_ids(5)
        mock_dataset = make_mock_id_embeddings_pair_dataset(
            mock_video_ids, (2,2))
        mock_precomputed_features = make_mock_precomputed_features(
            mock_video_ids)

        map_fn = encoder_datasets.replace_video_id_with_xpert_features_wrapper(
            mock_precomputed_features)

        output = list(iter(mock_dataset.map(map_fn)))

    def test_update_dataset_shape_wrapper(self):

    def test_zero_pad_expert_features(self):

    def test_get_precomputed_features(self):

    def test_sample_captions(self):