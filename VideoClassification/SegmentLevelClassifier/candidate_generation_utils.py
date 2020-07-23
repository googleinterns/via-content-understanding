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
    self.candidate_ids = [[]] * num_classes
    self.num_videos = 0
  
  def binary_search(self, candidate_probs, probability):
    """Binary search for the index in candidate_probs with the closest value to probability that is less than probability.

    Args:
      candidate_probs: list of values to be searched
      probability: value to search for
    """
    i = len(candidate_probs) // 2
    if len(candidate_probs) == 0:
      return 0
    elif probability < candidate_probs[i]:
      return i+self.binary_search(candidate_probs[i+1:], probability)
    else:
      return self.binary_search(candidate_probs[:i], probability)

  def sorted_append(self, class_index, video_index, probability, video_id):
    """Add video_index to the sorted list.

    Args:
      class_index: index of class to be appended to
      video_index: index of input video
      probability: output probability for class class_index
    """
    candidate_probs = self.candidate_probs[class_index]
    candidates = self.candidates[class_index]
    candidate_ids = self.candidate_ids[class_index]

    i = self.binary_search(candidate_probs, probability)

    self.candidate_probs[class_index] = candidate_probs[:i] + [probability] + candidate_probs[i:]
    self.candidates[class_index] = candidates[:i] + [video_index] + candidates[i:]
    self.candidate_ids[class_index] = candidate_ids[:i] + [video_id] + candidate_ids[i:]

    cand_probs = self.candidate_probs[class_index]
    for i in range(1,len(cand_probs)):
      assert cand_probs[i]
    print(self.candidate_probs[class_index])

  def sorted_insert(self, class_index, video_index, probability, video_id):
    """Add video_index to the sorted list, while removing the min.

    Args:
      class_index: index of class to be appended to
      video_index: index of input video
      probability: output probability for class class_index
    """
    #Remove min
    candidate_probs = self.candidate_probs[class_index][1:]
    candidates = self.candidates[class_index][1:]
    candidate_ids = self.candidate_ids[class_index][1:]

    i = self.binary_search(candidate_probs, probability)
    
    self.candidate_probs[class_index] = candidate_probs[:i] + [probability] + candidate_probs[i:]
    self.candidates[class_index] = candidates[:i] + [video_index] + candidates[i:]
    self.candidate_ids[class_index] = candidate_ids[:i] + [video_id] + candidate_ids[i:]

  def add_data(self, video_index, video_id, output_probs):
    """Add a datapoint to be sorted for the candidate generation.

    Args:
      video_index: index denoting the video number within the original dataset
      video_id: video id attained from the data itself
      output_probs: probability output from the video level model. Tensor of shape (video_num_classes,)
    """
    self.num_videos += 1

    #Go through each class and update candidate list if necessary.
    for class_index in range(self.num_classes):
      true_class_index = self.class_indices[class_index]
      probability = output_probs[true_class_index]
      if len(self.candidates[class_index]) < self.k:
        self.sorted_append(class_index, video_index, probability, video_id)
      elif output_probs[true_class_index] > self.candidates[class_index][0]:
        self.sorted_insert(class_index, video_index, probability, video_id)

  def find_candidates():
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
