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

Utils used to store data for candidate generation
"""
import bisect
import pandas as pd

class ProbabilityHolder:
  """Keeps track of the topk examples per class. Used for efficient candidate generation.
  """

  def __init__(self, class_csv, k):
    """Initialize an instance of ProbabilityHolder.

    Args:
      class_csv: path to csv file containing the indices of the classes used, out of the 3.8k output classes from the video-level classifier.
      k: number of examples to retain per class
    """
    class_dataframe = pd.read_csv(class_csv, index_col=0)
    class_indices = class_dataframe.index.tolist()
    num_classes = len(class_indices)

    self.num_classes = num_classes
    self.k = k
    self.class_indices = class_indices
    self.candidates = [[] for i in range(num_classes)]
    self.candidate_probs = [[] for i in range(num_classes)]
    self.num_videos = 0

  def sorted_append(self, class_index, probability, video_id):
    """Add video_index to the sorted list.

    Args:
      class_index: index of class to be appended to
      video_index: index of input video
      probability: output probability for class class_index
    """
    candidate_probs = self.candidate_probs[class_index]
    candidates = self.candidates[class_index]
    i = bisect.bisect(candidate_probs, probability)
    self.candidate_probs[class_index].insert(i, probability)
    self.candidates[class_index].insert(i, video_id)

  def sorted_insert(self, class_index, probability, video_id):
    """Add video_index to the sorted list, while removing the min.

    Args:
      class_index: index of class to be appended to
      video_index: index of input video
      probability: output probability for class class_index
    """
    #Remove min
    candidate_probs = self.candidate_probs[class_index][1:]
    candidates = self.candidates[class_index][1:]
    i = bisect.bisect(candidate_probs, probability)
    candidate_probs.insert(i, probability)
    candidates.insert(i, video_id)
    self.candidate_probs[class_index] = candidate_probs
    self.candidates[class_index] = candidates

  def add_data(self, video_id, output_probs):
    """Add a datapoint to be sorted for the candidate generation.

    Args:
      video_id: video id attained from the data itself. Must be hashable. If tensor, do tensor.ref()
      output_probs: probability output from the video level model. Tensor of shape (video_num_classes,)
    """
    self.num_videos += 1
    #Go through each class and update candidate list if necessary.
    for class_index in range(self.num_classes):
      true_class_index = self.class_indices[class_index]
      probability = output_probs[true_class_index]
      if len(self.candidates[class_index]) < self.k:
        self.sorted_append(class_index, probability, video_id)
      elif probability > self.candidate_probs[class_index][0]:
        self.sorted_insert(class_index, probability, video_id)

  def find_candidates(self):
    """Amass candidates.

    Returns:
      candidates: list of lists where each inner list designates the classes that example was chosen for. len of return value == self.num_videos
    """
    candidates = {}
    for class_index in range(len(self.candidates)):
      for video_candidate in self.candidates[class_index]:
        if video_candidate not in candidates.keys():
          candidates[video_candidate] = []
        candidates[video_candidate].append(class_index)
    return candidates
