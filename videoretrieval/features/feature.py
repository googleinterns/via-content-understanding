
class Feature:
	"""Wrapper class for features outputted by expert models."""
	def __init__(self, feature_name):
		self.feature_name = feature_name

	@property
	def name(self):
		return self.feature_name
	