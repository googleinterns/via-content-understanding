# Segment Level Classification

An implementation of the [cross class relevance learning](https://arxiv.org/abs/1911.08548) model.

To train the given model, the input pipeline is as follows:
dataset_split.py -> class_feature_generation.py -> train.py

To evaluate the model, the pipeline is:
candidate_generation.py -> dataset_split.py -> class_feature_generation.py -> combine_dataset.py -> evaluate.py