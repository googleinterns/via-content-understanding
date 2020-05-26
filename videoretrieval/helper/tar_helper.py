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

"""
import tarfile 

def generate_files_from_tar(tar_path, files_to_extract):
    """Extracts files from the given tar to the disk.


    parameters:
        tar_path: the path to the tar file
        files_to_extract: a list of tuples, where the first element is the
            path of the file inside the dar, and the second element is the path
            we'll export the file to.

        """

    tarfile_object = tarfile.open(tar_path)

    for path_inside_tar in files_to_extract:
        yield tarfile_object.extractfile(path_inside_tar)