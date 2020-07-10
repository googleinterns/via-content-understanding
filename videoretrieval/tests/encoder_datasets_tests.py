
import unittest
from train import encoder_datasets
import tensorflow as tf
import numpy as np

MOCK_TENSOR_SHAPE = (2, 2) 
MOCK_TENSOR_DATA = tf.zeros(MOCK_TENSOR_SHAPE)
NUM_MOCK_VIDEOS = 5
MOCK_CAPTIONS_PER_VIDEO = 20

def make_mock_video_ids(num_videos):
    video_ids = []

    for video_num in range(num_videos):
        video_ids.append(f"video{video_num}")

    return video_ids

def mock_id_embeddings_pair_dataset_generator(
    video_ids, mock_tensor_shape, mock_embeddings_per_video):
    for video_id in video_ids:
        for _ in range(mock_embeddings_per_video):
            yield (video_id, tf.zeros(mock_tensor_shape))

def make_mock_id_embeddings_pair_dataset(
    video_ids, mock_tensor_shape, mock_embeddings_per_video=1):
    
    dataset_generator = lambda: mock_id_embeddings_pair_dataset_generator(
        video_ids, mock_tensor_shape, mock_embeddings_per_video)

    return tf.data.Dataset.from_generator(dataset_generator, 
        (tf.string, tf.float32))

def make_mock_precomputed_features(video_ids):
    mock_precomputed_features_available = {}
    mock_precomputed_features_half_missing = {}

    for index, video_id in enumerate(video_ids):
        mock_precomputed_features_available[video_id] = (
            index + MOCK_TENSOR_DATA, True)
        mock_precomputed_features_half_missing[video_id] = (
            -1 * index + MOCK_TENSOR_DATA, index % 2 == 0)

    return [
        mock_precomputed_features_available,
        mock_precomputed_features_half_missing]

class TestEncoderDatasetsFunctions(unittest.TestCase):

    def test_replacing_video_id_with_expert_features(self):
        """Tests the replace_video_id_with_expert_features_wrapper function."""
        mock_video_ids = make_mock_video_ids(NUM_MOCK_VIDEOS)

        mock_dataset = make_mock_id_embeddings_pair_dataset(
            mock_video_ids, MOCK_TENSOR_SHAPE)

        mock_precomputed_features = make_mock_precomputed_features(
            mock_video_ids)

        map_fn = encoder_datasets.replace_video_id_with_expert_features_wrapper(
            mock_precomputed_features)

        output = list(iter(mock_dataset.map(map_fn)))

        for video_id, expert_features, _, missing_modalities in output:
            video_id = video_id.numpy().decode("utf-8")

            for feature_index, (feature, missing) in enumerate(zip(
                expert_features, missing_modalities)):

                expected_feature, expected_missing = \
                    mock_precomputed_features[feature_index][video_id]
                
                self.assertTrue(np.array_equal(
                    feature.numpy(), expected_feature.numpy()))
                self.assertEqual(missing, expected_missing)


    def test_update_dataset_shape_wrapper(self):
        pass

    def test_zero_pad_expert_features(self):
        pass

    def test_sample_captions(self):
        """Tests the sample captions method of encoder datasets."""
        mock_video_ids = make_mock_video_ids(NUM_MOCK_VIDEOS)

        mock_dataset = make_mock_id_embeddings_pair_dataset(
            mock_video_ids, MOCK_TENSOR_SHAPE, MOCK_CAPTIONS_PER_VIDEO)

        mock_dataset = list(iter(encoder_datasets.sample_captions(
            mock_dataset, MOCK_CAPTIONS_PER_VIDEO)))

        for expected_value, dataset_value in zip(mock_video_ids, mock_dataset):
            video_id, _ = dataset_value

            self.assertEqual(expected_value, video_id)


