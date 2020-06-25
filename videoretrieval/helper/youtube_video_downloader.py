""" Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Module for downloading youtube videos to disk by url.
"""

import pytube

def get_stream_to_download(video_streams):
    """Gets the stream to download from the provided video streams.

    arguments:
        video_streams: a video streams object provided by a video object from
            pytube.
    """

    mp4_streams = video_streams.filter(file_extension="mp4")

    stream = mp4_streams.get_by_resolution("360p")

    if stream is None:
        stream = mp4_streams.get_highest_resolution()

    return stream

def download_video(video_url, output_path, filename):
    """Downloads a youtube video given a video url.

    arguments:
        video_url: a url string of the youtube video
        output_path: the directory to save the file in
        filename: the name of the file to save the video as
    """

    video_object = pytube.YouTube(video_url)

    stream = get_stream_to_download(video_object.streams)

    if stream is not None:
        stream.download(ouptut_path=output_path, filename=filename)
