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

Tests the msrvtt dataset code.
"""
from datasets import msrvtt 
import unittest

class TestMSRVTTMetadata(unittest.TestCase):
    """Tests parts of the MSRVTTDataset class."""

    def test_match_videos_and_captions(self):
        """Test the match_videos_and_captions function."""

        videos = [{
            "video_id": "video001",
            "arbitrary_field": 0,
            "split": "train"
        }, {
            "video_id": "video002",
            "arbitrary_field": 10,
            "split": "validation"
        }]

        captions = [{
            "video_id": "video001",
            "second_arbitrary_field": 5,
            "caption": "Caption #1"
        }, {
            "video_id": "video001",
            "second_arbitrary_field": 6,
            "caption": "Caption #2"
        }, {
            "video_id": "video002",
            "second_arbitrary_field": 7,
            "caption": "Caption #3"
        }]

        output = msrvtt.metadata.match_videos_and_captions(videos, captions)

        expected_output = {
            "train": [{
                "video_id": "video001",
                "arbitrary_field": 0,
                "split": "train",
                "captions": ["Caption #1", "Caption #2"]
            }],
            "validation": [{
                "video_id": "video002",
                "arbitrary_field": 10,
                "split": "validation",
                "captions": ["Caption #3"]
            }]
        }

        self.assertEquals(output, expected_output)

if __name__ == "__main__":
    unittest.main()