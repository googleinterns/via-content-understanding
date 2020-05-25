
from abc import ABC as AbstractClass
from abc import abstractmethod
import pathlib

class BaseExpert(AbstractClass):

	@property
	@abstractmethod
	def name(self):
		pass
	
	@property
	@abstractmethod
	def embedding_shape(self):
		pass