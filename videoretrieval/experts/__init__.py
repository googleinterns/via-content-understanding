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

__init__.py for experts package. Imports and initializes expert models.
"""

from .action import I3D, R2P1D
from .objects import ResNext101, SeNet154
from .speech import SpeechExpert
from .ocrexpert import OCRExpert
from .scene import DenseNet161

i3d = I3D()
r2p1d = R2P1D()

resnext = ResNext101()
senet = SeNet154()

speech_expert = SpeechExpert()

ocr = OCRExpert()
densenet = DenseNet161()