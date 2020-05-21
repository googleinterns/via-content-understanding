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

Functions to download, extract, and process precomputed features.
"""

import tarfile
from helper import file_downloader
from . import constants

features_to_extract = {
    "data/MSRVTT/structured-symlinks/aggregated_audio_feats/Audio_MSRVTT_new.pickle"
    "data/MSRVTT/structured-symlinks/aggregated_face_feats/facefeats-avg.pickle"
    "data/MSRVTT/structured-symlinks/aggregated_i3d_25fps_256px_stride25_offset0_inner_stride1/i3d-avg.pickle"
    "data/MSRVTT/structured-symlinks/aggregated_imagenet_25fps_256px_stride1_offset0/resnext101_32x48d-avg.pickle"
    "data/MSRVTT/structured-symlinks/aggregated_imagenet_25fps_256px_stride1_offset0/senet154-avg.pickle"
    "data/MSRVTT/structured-symlinks/aggregated_ocr_feats/ocr-raw.pickle"
    "data/MSRVTT/structured-symlinks/aggregated_r2p1d_30fps_256px_stride32_offset0_inner_stride1/r2p1d-ig65m-avg.pickle"
    "data/MSRVTT/structured-symlinks/aggregated_scene_25fps_256px_stride1_offset0/densenet161-avg.pickle"
    "data/MSRVTT/structured-symlinks/aggregated_speech/speech-w2v.pickle"
}

def download_features_tar():
    file_downloader.download_by_url(
        constants.features_tar_url,
        constants.features_tar_path
    )

def extract_tar_features():
    features_tar = tarfile.open(constants.features_tar_path)

    feature_vectors = {}

    for pickle_pa