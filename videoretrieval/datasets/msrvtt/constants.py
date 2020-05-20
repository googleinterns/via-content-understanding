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

Defines paths and urls for msrvtt dataset.
"""

from pathlib import Path

train_val_zip_url = "http://ms-multimedia-challenge.com/static/resource/train_val_annotation.zip"
test_data_url = "http://ms-multimedia-challenge.com/static/resource/test_videodatainfo.json"

base_dir = Path("./downloaded_data/datasets/msr_vtt/")

train_val_zip_path = base_dir / "train_val_zip_metadata.zip"
train_val_json_filename = "train_val_videodatainfo.json"
test_json_path = base_dir / "test_metadata.json"
processed_metadata_path = base_dir / "metadata.json"