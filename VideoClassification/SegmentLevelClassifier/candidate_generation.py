import candidate_generation_utils as utils
import pandas as pd
import readers
import reader_utils
import tensorflow as tf


def load_model(model_dir):
  model_generator = model_lib.VideoClassifier(num_clusters, video_input_shape, audio_input_shape, fc_units=fc_units, num_classes=data_reader.num_classes, num_mixtures=num_mixtures, iterations=iterations)
  
  model = model_generator.build_model(input_shape, batch_size)

  model.load_weights(model_path)

  return model

def save_data(new_data_dir, input_dataset, candidates):
  """Save data as TFRecords Datasets in new_data_dir.

  Args:
    new_data_dir: string giving the directory to save TFRecords Files
    input_dataset: original dataset before candidate generation
    candidates: list of lists where each inner list contains the class indices that the corresponding input data is a candidate for. len(candidates) == len(input_dataset)
  """


def generate_candidates(input_dataset, model, k, class_csv):
  """Generate top k candidates per class.

  Args:
    input_dataset: dataset to be chosen from
    model: model used to rank data
    k: number of candidates per class
    class_csv: path to csv file containing the indices of the classes used, out of the 3.8k output classes from the video-level classifier.
  Returns:
    candidates: list of lists where each inner list contains the class indices that the corresponding input data is a candidate for. len(candidates) == len(input_dataset)
  """
  probability_holder = utils.PROBABILITY_HOLDER(class_csv, k)
  
  video_index = 0
  for video in input_dataset:
    probability_holder.add_data(video_index, model.predict(video))
    video_index += 1

  return probability_holder.find_candidates()

  

if __name__ == "__main__":
  reader = readers.YT8MSegmentsDataset()
  input_dataset = reader_utils.load_dataset(reader, "~/data/segments/validation")

  model = load_model("~/model.h5")
  candidates = generate_candidates(input_dataset, model, 100)

  save_data(new_data_dir="~/data/segments/new_validation", input_dataset, candidates)

