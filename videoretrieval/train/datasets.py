"""Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

def build_dataset_from_cached_features(dataset, experts, language_model):
	train_ids, valid_ids, test_ids = dataset.train_valid_test_ids

	train_data = {}
	valid_data = []
	test_data = []

	for video_id, caption in dataset.video_captions:
		if video_id in train_ids:
			train_data.append()
