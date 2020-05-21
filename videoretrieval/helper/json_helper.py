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

Module for handling file i/o with json data.
"""

import json

def load_json_from_file(file_path):
    """Loads and returns json data from the file at file_path.
    
    arguments:
        file_path: a path of the json file to load.
    """

    with open(file_path, "rb") as file:
        result = json.load(file)

    return result

def save_json_to_file(json_data, file_path):
    """Saves the data in json_data as json in a file at file_path.

    arguments:
        json_data: data that can be converted to json.
        file_path: a path to the file to save the json.
    """

    with open(file_path, "w") as file:
        json.dump(json_data, file)
