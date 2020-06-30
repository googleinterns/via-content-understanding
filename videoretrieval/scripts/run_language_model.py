import train.language_models
import languagemodels

train.language_model.generate_and_cache_contextual_embeddings(
    languagemodels.openai_gpt, datasets.msrvtt_dataset)