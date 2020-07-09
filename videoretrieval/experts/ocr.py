"""Classes for managing OCR experts.

Copyright 2020 Google LLC

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

from base import BaseExpert

class OCRExpert(BaseExpert):
    """Implementation of the OCR expert class."""

    @property
    def name(self):
        return "ocr"
    
    @property
    def embedding_shape(self):
        return (49, 300)

    @property
    def constant_length(self):
        return False

    @property
    def max_frames(self):
        return 49
