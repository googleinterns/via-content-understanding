import pandas as pd

class PROBABILITY_HOLDER:
  """Keeps track of the topk examples per class. Used for efficient candidate generation.
  """

  def __init__(self, class_csv, k):
    """Initialize an instance of PROBABILITY_HOLDER.

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
    self.candidates = [[]] * num_classes
    self.candidate_probs = [[]] * num_classes
    self.num_videos = 0
  
  def binary_search(self, sorted_list, input):
    """Binary search for the index in candidate_probs with the closest value to probability that is less than or equal to probability.

    Args:
      sorted_list: list of values to be searched. Sorted in increasing order
      input: value to search for
    """
    low = 0
    high = len(sorted_list)-1
    while low <= high:
      mid = (low + high) // 2
      if sorted_list[mid] == input:
        return mid
      elif sorted_list[mid] < input:
        low = mid + 1
      else:
        high = mid - 1
    mid = (low + high) // 2
    return mid


  def sorted_append(self, class_index, probability, video_id):
    """Add video_index to the sorted list.

    Args:
      class_index: index of class to be appended to
      video_index: index of input video
      probability: output probability for class class_index
    """
    candidate_probs = self.candidate_probs[class_index]
    candidates = self.candidates[class_index]

    i = self.binary_search(candidate_probs, probability)

    self.candidate_probs[class_index] = candidate_probs[:i+1] + [probability] + candidate_probs[i+1:]
    self.candidates[class_index] = candidates[:i+1] + [video_id] + candidates[i+1:]

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

    i = self.binary_search(candidate_probs, probability)
    
    self.candidate_probs[class_index] = candidate_probs[:i+1] + [probability] + candidate_probs[i+1:]
    self.candidates[class_index] = candidates[:i+1] + [video_id] + candidates[i+1:]

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
        assert probability in self.candidate_probs[class_index]
        assert video_id in self.candidates[class_index]
      elif probability > self.candidate_probs[class_index][0]:
        min_id = self.candidates[class_index][0]
        min_probs = self.candidate_probs[class_index][0]
        self.sorted_insert(class_index, probability, video_id)
        assert probability in self.candidate_probs[class_index]
        assert video_id in self.candidates[class_index]
        assert min_probs not in self.candidate_probs[class_index]
        assert min_id not in self.candidates[class_index]

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

    for i in self.candidates:
      for j in i:
        assert j in candidates.keys()
    print(len(candidates))
    return candidates
