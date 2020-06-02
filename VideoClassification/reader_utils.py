import readers
import tensorflow as tf
from tensorflow.io import gfile


def get_reader(feature_names='rgb,audio', feature_sizes='1024,128'):
	# Convert feature_names and feature_sizes to lists of values.
	feature_names, feature_sizes = GetListOfFeatureNamesAndSizes(
			feature_names, feature_sizes)

	reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
																						feature_sizes=feature_sizes)
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
