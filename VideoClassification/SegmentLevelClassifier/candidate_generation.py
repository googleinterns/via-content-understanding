import candidate_generation_utils as utils
import model as model_lib
import numpy as np
import os
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

def add_candidate_content(context, candidates):
  """Add the tensor for the classes this particular video is a candidate for.

  context: context of the video
  candidates: dictionary of candidates. Key is video id and value is list of candidate classes
  """
  print(candidates)
  video_id = tf.convert_to_tensor(context["id"])[0].ref()
  if video_id in candidates.keys():
    context["candidate_labels"] = tf.convert_to_tensor(candidates[video_id])
  else:
    context["candidate_labels"] = tf.convert_to_tensor([])
  print(context)
  return context

def convert_labels(labels, class_csv="vocabulary.csv"):
  """Convert labels from range [0,3861] to range [0,1000]

  labels: Tensor of labels to be converted
  class_csv: csv file containing conversion details
  """
  class_dataframe = pd.read_csv(class_csv, index_col=0)
  class_indices = class_dataframe.index.tolist()
  class_indices = np.array(class_indices)

  labels = labels.numpy()
  new_labels = []
  for label in labels:
    check = np.nonzero(class_indices == label)
    if np.any(check):
      new_label = check[0].tolist()[0]
      new_labels.append(new_label)
  return tf.convert_to_tensor(new_labels)

def convert_to_feature(item, type):
  """Convert item to FeatureList.

  item: item to be converted
  type: string denoting the type of item. Can be "float", "byte", or "int"
  """
  if type == "float":
    item = tf.train.FloatList(value=item)
    item = tf.train.Feature(float_list=item)
  elif type == "byte":
    item = tf.train.BytesList(value=item)
    item = tf.train.Feature(bytes_list=item)
  elif type == "int":
    item = tf.train.Int64List(value=item)
    item = tf.train.Feature(int64_list=item)
  else:
    print("Invalid type entered for converting feature")
  return item

def serialize_features(features):
  """Serialize features.

  features: features of the video
  """
  audio = features["audio"][0].numpy().tostring()
  rgb = features["rgb"][0].numpy().tostring()
  audio = convert_to_feature([audio], "byte")
  rgb = convert_to_feature([rgb], "byte")
  features = {"audio": tf.train.FeatureList(feature=[audio]), "rgb": tf.train.FeatureList(feature=[rgb])}
  features = tf.train.FeatureLists(feature_list=features)
  return features

def serialize_context(context):
  """Serialize context.

  context: context of the video
  """
  video_id = tf.convert_to_tensor(context["id"])[0]
  labels = context["labels"].values
  segment_labels = context["segment_labels"].values
  segment_start_times = context["segment_start_times"].values
  segment_scores = context["segment_scores"].values
  #labels =  convert_labels(labels)
  #segment_labels = convert_labels(segment_labels)

  print(video_id)
  print(labels)
  print(segment_labels)
  print(segment_start_times)
  print(segment_scores)

  context["id"] = convert_to_feature([video_id.numpy()], "byte")
  context["labels"] = convert_to_feature(labels.numpy(), "int")
  context["segment_labels"] = convert_to_feature(segment_labels.numpy(), "int")
  context["segment_start_times"] = convert_to_feature(segment_start_times.numpy(), "int")
  context["segment_scores"] = convert_to_feature(segment_scores.numpy(), "float")

  context = tf.train.Features(feature=context)
  return context

def serialize_video(context, features):
  """Serialize video from context and features.

  context: context of the video
  features: features of the video
  """
  features = serialize_features(features)
  context = serialize_context(context)
  example = tf.train.SequenceExample(feature_lists=features, context=context)
  return example.SerializeToString()

def save_data(new_data_dir, input_dataset, candidates, file_type="validate", shard_size=5):
  """Save data as TFRecords Datasets in new_data_dir.

  Args:
    new_data_dir: string giving the directory to save TFRecords Files
    input_dataset: original dataset before candidate generation
    candidates: list of lists where each inner list contains the class indices that the corresponding input data is a candidate for. len(candidates) == len(input_dataset)
  """
  shard_counter = 0
  shard_number = 0
  shard = []
  for video in input_dataset:
    context = video[0]
    features = video[1]
    context = add_candidate_content(context, candidates)
    serialized_video = serialize_video(context, features)
    shard.append(serialized_video)
    shard_counter += 1
    assert False
    if shard_counter == shard_size:
      shard = tf.convert_to_tensor(shard)
      shard_dataset = tf.data.Dataset.from_tensor_slices(shard)
      file_name = file_type + str(shard_number)
      file_path = os.path.join(new_data_dir, '%s.tfrecord' % file_name)
      writer = tf.data.experimental.TFRecordWriter(file_path)
      writer.write(shard_dataset)
      shard_counter = 0
      shard_number += 1
      shard = []

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
  video_num = 0
  input_dataset = tfds.as_numpy(input_dataset)
  for video in input_dataset:
    print(f"Processing video number {video_num}")
    video_id = tf.convert_to_tensor(video[0])[0].ref()
    video_input = tf.convert_to_tensor(video[1])
    probability_holder.add_data(video_id, model.predict(video_input)[0])
    video_num += 1
  return probability_holder.find_candidates()

if __name__ == "__main__":
  #do candidate gen and keep track of video_ids. Then, pass list of (video_id, class_list) pairs to dataset to then add them while loading data. 
  #then simply parse that like Ryan and write it in shards
  video_reader = readers.VideoDataset()
  input_dataset = video_reader.get_dataset("/home/conorfvedova_google_com/data/segments/validation", batch_size=1, type="validate")

  model = load_model("../model_weights.h5")

  candidates = generate_candidates(input_dataset, model, 10, "vocabulary.csv")

  segment_reader = readers.PreprocessingDataset()
  input_dataset = segment_reader.get_dataset("/home/conorfvedova_google_com/data/segments/validation", batch_size=1, type="validate")
  save_data("/home/conorfvedova_google_com/data/segments/candidate_validation", input_dataset, candidates)
