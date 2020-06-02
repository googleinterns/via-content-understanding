import cache
import tensorflow as tf
import numpy as np

def replace_video_id_with_expert_features_wrapper(precomputed_features):
    output_shape = len(precomputed_features) * (tf.float32,)

    def get_expert_features(video_id_encoded):
        expert_features = []

        video_id = video_id_encoded.decode("utf-8")

        for feature_dict in precomputed_features:
            expert_features.append(feature_dict[video_id].astype(np.float32))

        return expert_features

    def wrapper(video_id, ids, source):
        expert_features = tf.numpy_function(
            get_expert_features, [video_id], output_shape)

        return (tuple(expert_features), ids)

    return wrapper

def match_cached_embeddings_with_experts(precomputed_features, *datasets):
    map_fn = replace_video_id_with_expert_features_wrapper(precomputed_features)

    return [(dataset
        .map(map_fn)#, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .prefetch(tf.data.experimental.AUTOTUNE)) for dataset in datasets]

def get_precomputed_features(source_dataset, experts):
    precomputed_features = []

    for expert in experts:
        precomputed_features.append(
            cache.get_cached_features_by_expert_and_dataset(
                source_dataset, expert))

    return precomputed_features

def generate_encoder_datasets(language_model, source_dataset, experts):
    train_ds = cache.get_cached_language_model_embeddings(
        source_dataset, language_model, "train")

    valid_ds = cache.get_cached_language_model_embeddings(
        source_dataset, language_model, "valid")

    test_ds = cache.get_cached_language_model_embeddings(
        source_dataset, language_model, "test")

    precomputed_features = get_precomputed_features(source_dataset, experts)

    return match_cached_embeddings_with_experts(
        precomputed_features, train_ds, valid_ds, test_ds)
