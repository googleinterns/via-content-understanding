import candidate_generation_utils
import tensorflow as tf
import unittest

class TestCandidateGeneration(unittest.TestCase):
  """Class used to test the candidate generation utils defined in candidate_generation_utils.py"""

  def test_sorted_append(self):
    prob_holder = ProbabilityHolder("vocabulary.csv", 50)
    candidates = prob_holder.candidates.copy()
    candidate_probs = prob_holder.candidate_probs.copy()
    
    prob_holder.sorted_append(0, 0.1, 0)
    prob_holder.sorted_append(0, 0.5, 1)
    prob_holder.sorted_append(0, 0.2, 2)
    prob_holder.sorted_append(0, 0.2, 3)
    prob_holder.sorted_append(999, 0.6, 2)

    candidates[0] = [0, 2, 3, 1]
    candidate_probs[0] = [0.1, 0.2, 0.2, 0.5]

    candidates[999] = [2]
    candidate_probs[999] = [0.6]

    self.assertEqual(candidates, prob_holder.candidates)
    self.assertEqual(candidate_probs, prob_holder.candidate_probs)

  def test_sorted_insert(self):
    prob_holder = ProbabilityHolder("vocabulary.csv", 2)
    candidates = prob_holder.candidates.copy()
    candidate_probs = prob_holder.candidate_probs.copy()
    
    prob_holder.sorted_append(0, 0.1, 0)
    prob_holder.sorted_append(0, 0.5, 1)
    prob_holder.sorted_append(0, 0.2, 2)
    prob_holder.sorted_insert(0, 0.2, 3)
    prob_holder.sorted_insert(0, 0.2, 4)
    prob_holder.sorted_insert(0, 0.9, 5)

    candidates[0] = [4, 1, 5]
    candidate_probs[0] = [0.2, 0.5, 0.9]

    self.assertEqual(candidates, prob_holder.candidates)
    self.assertEqual(candidate_probs, prob_holder.candidate_probs)

    def test_find_candidates(self):
      prob_holder = ProbabilityHolder("vocabulary.csv", 2)
      
      prob_holder.sorted_append(0, 0.1, 0)
      prob_holder.sorted_append(0, 0.5, 1)
      prob_holder.sorted_append(0, 0.2, 2)
      prob_holder.sorted_insert(0, 0.2, 3)
      prob_holder.sorted_insert(0, 0.2, 4)
      prob_holder.sorted_insert(0, 0.9, 5)
      prob_holder.sorted_append(1, 0.5, 1)

      candidates = prob_holder.find_candidates()

      test_candidates = {}
      test_candidates[4] = [0]
      test_candidates[1] = [0, 1]
      test_candidates[5] = [0]

      self.assertEqual(candidates, test_candidates)

if __name__ == "__main__":
  unittest.main()