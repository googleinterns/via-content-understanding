# Segment Level Classification

An implementation of the [cross class relevance learning](https://arxiv.org/abs/1911.08548) model.

The data used for this model can be found at [YouTube-8M Segments Dataset](https://research.google.com/youtube8m/index.html).

To train the given model, the input pipeline is as follows:
dataset_split.py -> class_feature_generation.py -> train.py

Example Commands for training:
```
python dataset_split.py --input_dir=data/train --write_dir=data/split_train --pipeline_type=train
python class_feature_generation.py --input_dir=data/split_train --write_dir=data/input_train_data --comparison_directory=data/split_train --pipeline_type=train
python train.py --input_dir=data/input_train_data --model_path=model_weights.h5 --logs_dir=logs
```

To evaluate the model, the pipeline is:
candidate_generation.py -> dataset_split.py -> class_feature_generation.py -> combine_dataset.py -> evaluate.py

Example Commands for evaluation:
```
python candidate_generation.py --input_dir=data/test --model_weights_path=model_weights.h5 --file_type_name=test --write_dir=data/test_candidate_gen
python dataset_split.py --input_dir=data/test_candidate_gen --write_dir=data/split_test --pipeline_type=test
python class_feature_generation.py --input_dir=data/split_test --write_dir=data/input_test_data --comparison_directory=data/split_train --pipeline_type=test
python combine_dataset.py --input_dir=data/input_test_data --write_dir=data/finalized_test_data
python evaluate.py --input_dir=data/finalized_test_data --model_weights_path=model_weights.h5
```