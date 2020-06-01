import readers
import tensorflow as tf
from tensorflow.io import gfile

def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1):
  """Creates the section of the graph which reads the training data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the training data. Set to 'None' to
      run indefinitely.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("train_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find training files. data_pattern='" +
                    data_pattern + "'.")
    filename_queue = tf.train.string_input_producer(files,
                                                    num_epochs=num_epochs,
                                                    shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    return tf.train.shuffle_batch_join(training_data,
                                       batch_size=batch_size,
                                       capacity=batch_size * 5,
                                       min_after_dequeue=batch_size,
                                       allow_smaller_final_batch=True,
                                       enqueue_many=True)

def get_reader(feature_names='rgb,audio', feature_sizes='1024,128', segment_labels=False):
  # Convert feature_names and feature_sizes to lists of values.
  feature_names, feature_sizes = GetListOfFeatureNamesAndSizes(
      feature_names, feature_sizes)

  reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                            feature_sizes=feature_sizes,
                                            segment_labels=segment_labels)
  return reader

def GetListOfFeatureNamesAndSizes(feature_names, feature_sizes):
  """Extract the list of feature names and the dimensionality of each feature

     from string of comma separated values.

  Args:
    feature_names: string containing comma separated list of feature names
    feature_sizes: string containing comma separated list of feature sizes

  Returns:
    List of the feature names and list of the dimensionality of each feature.
    Elements in the first/second list are strings/integers.
  """
  list_of_feature_names = [
      feature_names.strip() for feature_names in feature_names.split(",")
  ]
  list_of_feature_sizes = [
      int(feature_sizes) for feature_sizes in feature_sizes.split(",")
  ]
  if len(list_of_feature_names) != len(list_of_feature_sizes):
    logging.error("length of the feature names (=" +
                  str(len(list_of_feature_names)) + ") != length of feature "
                  "sizes (=" + str(len(list_of_feature_sizes)) + ")")

  return list_of_feature_names, list_of_feature_sizes
