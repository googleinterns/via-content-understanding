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

Module for downloading files to disk by url.
"""

import requests

def download_by_url(url, output_path):
    """Downloads a file given its url.

    arguments:
        url: the file's location
        output_path: path to write the file to
    """

    request = requests.get(url)

    with open(output_path, "wb") as file:
        file.write(request.content)
