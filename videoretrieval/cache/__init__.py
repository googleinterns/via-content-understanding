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

__init__.py for the cache package.
"""

from .features_cache import cache_features_by_expert_and_dataset, \
	get_cached_features_by_expert_and_dataset 

from .language_model_cache import cache_language_model_embeddings, \
	get_cached_language_model_embeddings, cache_language_model_encodings, \
	get_cached_language_model_encodings
