import cache

def match_cached_embeddings_with_experts(precomputed_features, *datasets):

def get_precomputed_features(dataset, experts):
    precomputed_features = []

    for expert in experts:
        precomputed_features.append(
            cache.get_cached_features_by_expert_and_dataset(
                expert, source_dataset))

    return precomputed_features

def generate_encoder_dataset(language_model, source_dataset, experts):
    train_ds = get_cached_language_model_embeddings(
        source_dataset, language_model, "train")

    valid_ds = get_cached_language_model_embeddings(
        source_dataset, language_model, "valid")

    test_ds = get_cached_language_model_embeddings(
        source_dataset, language_model, "test")

    precomputed_features = get_precomputed_features(source_dataset, experts)

    train_ds = get_expert_features(train_ds)
    valid_ds = get_expert_features(valid_ds)
    test_ds = get_expert_features(test_ds)
