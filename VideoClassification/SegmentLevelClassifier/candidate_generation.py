import candidate_generation_utils as utils
import model as model_lib
import pandas as pd
import readers
import reader_utils
import tensorflow as tf
import tensorflow_datasets as tfds


def load_model(model_path, num_clusters=256, batch_size=80, random_frames=True, num_mixtures=2, fc_units=1024, iterations=300, num_classes=3862):
  video_input_shape = (batch_size, iterations, 1024)
  audio_input_shape = (batch_size, iterations, 128)
  input_shape = (iterations, 1152)

  model_generator = model_lib.VideoClassifier(num_clusters, video_input_shape, audio_input_shape, fc_units=fc_units, num_classes=num_classes, num_mixtures=num_mixtures, iterations=iterations)
  
  model = model_generator.build_model(input_shape, batch_size)

  model.load_weights(model_path)

  return model

def save_data(new_data_dir, input_dataset):
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
  input_dataset = tfds.as_numpy(input_dataset)
  for video in input_dataset:
    video_id = tf.convert_to_tensor(video[0])[0].eval()
    video_input = tf.convert_to_tensor(video[1])
    print(video_id)
    probability_holder.add_data(video_index, video_id, model.predict(video_input)[0])
    video_index += 1

  return probability_holder.find_candidates()

  

if __name__ == "__main__":
  #do candidate gen and keep track of video_ids. Then, pass list of (video_id, class_list) pairs to dataset to then add them while loading data. 
  #then simply parse that like Ryan and write it in shards
  video_reader = readers.VideoDataset()
  input_dataset = video_reader.get_dataset("/home/conorfvedova_google_com/data/segments/validation", batch_size=1, type="validate")

  model = load_model("../model_weights.h5")

  candidates = generate_candidates(input_dataset, model, 100, "vocabulary.csv")

  segment_reader = readers.PreprocessingDataset(candidates=candidates)
  input_dataset = segment_reader.get_dataset("/home/conorfvedova_google_com/data/segments/validation", batch_size=1, type="validate")

  save_data("~/data/segments/new_validation", input_dataset)
