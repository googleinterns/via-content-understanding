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
import experts

train_val_zip_url = "http://ms-multimedia-challenge.com/static/resource/train_val_annotation.zip"
test_data_url = "http://ms-multimedia-challenge.com/static/resource/test_videodatainfo.json"

base_dir = Path("./downloaded_data/datasets/msr_vtt/")

train_val_zip_path = base_dir / "train_val_zip_metadata.zip"
train_val_json_filename = "train_val_videodatainfo.json"
test_json_path = base_dir / "test_metadata.json"
processed_metadata_path = base_dir / "metadata.json"

features_tar_url = "http://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/features-v2/MSRVTT-experts.tar.gz"
features_tar_path = base_dir / "precomputed_features.tar"

precomputed_features_base_path = "data/MSRVTT/structured-symlinks/"

i3d_path = precomputed_features_base_path + \
	"aggregated_i3d_25fps_256px_stride25_offset0_inner_stride1/i3d-avg.pickle"
resnext_path = precomputed_features_base_path + \
	"aggregated_imagenet_25fps_256px_stride1_offset0/resnext101_32x48d-avg.pickle"
senet_path = precomputed_features_base_path + \
	"aggregated_imagenet_25fps_256px_stride1_offset0/senet154-avg.pickle"
speech_path = precomputed_features_base_path + \
	"aggregated_speech/speech-w2v.pickle"
r2p1d_path = precomputed_features_base_path + \
	"aggregated_r2p1d_30fps_256px_stride32_offset0_inner_stride1/r2p1d-ig65m-avg.pickle"
ocr_path = precomputed_features_base_path + \
	"aggregated_ocr_feats/ocr-raw.pickle"
densenet_path = precomputed_features_base_path + \
	"aggregated_scene_25fps_256px_stride1_offset0/densenet161-avg.pickle"
audio_path = precomputed_features_base_path + \
	"aggregated_audio_feats/Audio_MSRVTT_new.pickle"
face_path = precomputed_features_base_path + \
	"aggregated_face_feats/facefeats-avg.pickle"

audio_path = precomputed_features_base_path + \
	"aggregated_audio_feats/Audio_MSRVTT_new.pickle"

face_path = precomputed_features_base_path + \
	"aggregated_face_feats/facefeats-avg.pickle"

expert_to_features = {
	experts.i3d: i3d_path,
	experts.resnext: resnext_path,
	experts.senet: senet_path,
	experts.speech_expert: speech_path,
	experts.r2p1d: r2p1d_path,
	experts.ocr_expert: ocr_path,
	experts.densenet: densenet_path,
	experts.audio_expert: audio_path,
	experts.face_expert: face_path
}
