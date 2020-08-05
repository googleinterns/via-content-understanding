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

Functions to download, process, and save metadata for the msrvtt dataset.
"""
import os
from . import constants
from helper import file_downloader
from helper import json_helper
from zipfile import ZipFile

def download_and_load_metadata():
    """Downloads the metadata, processes it, saves it, and returns it."""

    download_dataset_video_metadata()
    
    dataset_metadata = generate_video_metadata()
    
    json_helper.save_json_to_file(
    	dataset_metadata,
    	constants.processed_metadata_path
    )

    remove_unnecessary_metadata()

    return dataset_metadata

def load_metadata():
	"""Loads the saved metadata."""
	return json_helper.load_json_from_file(constants.processed_metadata_path)

def get_dataset_metadata():
    """Returns the metadata provided by the dataset.

    returns:
        A tuple of (videos, sentences) where videos is a list of dicts where
        each dict has data about a video, and sentences is a list of dicts
        where each dict has data about a caption. 
    """ 
    train_val_metadata = json_helper.load_json_from_file(
        constants.base_dir / constants.train_val_json_filename)

    test_metadata = json_helper.load_json_from_file(constants.test_json_path)

    videos = train_val_metadata["videos"] + test_metadata["videos"]
    sentences = train_val_metadata["sentences"] + test_metadata["sentences"]

    return videos, sentences

def generate_video_metadata():
    """Maps the metadata from the dataset to a consumable format."""
    videos, sentences = get_dataset_metadata()

    return match_videos_and_captions(videos, sentences)


def match_videos_and_captions(videos, sentences):
    """Maps captions to videos and splits the data into train/valid/test.

    arguments:
        videos: a list of dicts, where each dict describes a video
        sentences: a list of dicts, where each dict describes a sentence

    returns:
        a dict that maps from split name to a list of videos, where each video
        is a dict that contains a field "url" that is the video's url, and a
        field "captions", a list of strings where each string is a possible
        caption of the video.
    """
    dataset_metadata = {}
    video_id_to_split_index_pair = {}

    for video in videos:
        split = video["split"]
        id_ = video["video_id"]

        if split not in dataset_metadata:
            dataset_metadata[split] = []

        video["captions"] = []
        video_id_to_split_index_pair[id_] = (
        	split, 
        	len(dataset_metadata[split])
        )

        dataset_metadata[split].append(video)

    for sentence in sentences:
        caption_text = sentence["caption"]
        video_id = sentence["video_id"]
        
        video_split, video_index = video_id_to_split_index_pair[video_id]
        video = dataset_metadata[video_split][video_index]

        video["captions"].append(sentence["caption"])

    return dataset_metadata

def download_dataset_video_metadata():
    """Downloads metadata for train/validation/test sets."""
    constants.base_dir.mkdir(parents=True, exist_ok=True)

    download_train_val_video_metadata()
    download_test_video_metadata()

def download_train_val_video_metadata():
    """Downloads the train/validation metadata and extracts the json."""
    file_downloader.download_by_url(
        url=constants.train_val_zip_url,
        output_path=constants.train_val_zip_path
    )

    extract_train_val_json()

def extract_train_val_json():
    """Extracts the train/validation json file from the enclosing zip."""
    train_val_zip = ZipFile(constants.train_val_zip_path)

    train_val_zip.extract(constants.train_val_json_filename, constants.base_dir)

def download_test_video_metadata():
    """Downloads the metadata for videos in the test set."""
    file_downloader.download_by_url(
        url=constants.test_data_url,
        output_path=constants.test_json_path
    )

def remove_unnecessary_metadata():
    """Removes the raw metadata from the file system."""
    os.remove(constants.train_val_zip_path)
    os.remove(constants.base_dir / constants.train_val_json_filename)
    os.remove(constants.test_json_path)
