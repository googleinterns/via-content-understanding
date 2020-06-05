import readers
import tensorflow as tf
from tensorflow.io import gfile


def get_reader(num_samples, random_frames, feature_names='rgb,audio', feature_sizes='1024,128'):
	# Convert feature_names and feature_sizes to lists of values.
	feature_names, feature_sizes = GetListOfFeatureNamesAndSizes(feature_names, feature_sizes)

	reader = readers.YT8MFrameFeatureDataset(feature_names=feature_names, feature_sizes=feature_sizes, num_samples=num_samples, random_frames=random_frames)

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

def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
  """Dequantize the feature from the byte format to the float format.

  Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    A float vector which has the same shape as feat_vector.
  """
  assert max_quantized_value > min_quantized_value
  quantized_range = max_quantized_value - min_quantized_value
  scalar = quantized_range / 255.0
  bias = (quantized_range / 512.0) + min_quantized_value
  return feat_vector * scalar + bias