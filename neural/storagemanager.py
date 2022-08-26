from datetime import datetime
from pathlib import Path
from collections import namedtuple

dataset_pair = namedtuple("dataset_pair", ["train", "test"])

class StorageManager():
	""" Consistent management of directories and locations for neural models """

	def __init__(self, models_dir = "train", data_dir = "train", tensorboard_logdir="runs"):
		self.models_dir = models_dir
		self.data_dir = data_dir
		self.tbdir = tensorboard_logdir
	
	def sessions(self):
		""" Returns a list of all sessions in the models directory """
		return list(Path(self.models_dir).glob("v1_*"))
	
	def models(self, session):
		""" Returns a list of all models in the session """
		return list(Path(session).glob("*.pt"))
	
	def last_model(self):
		""" Returns the path to the last model in the models directory """
		return self.models(self.sessions()[-1])[-1]
	
	def datasets(self):
		""" Returns a list of all datasets in the data directory """
		# this is a little weak: if the *_train and *_test are not always in pairs,
		# it will mess up all the following pairings
		return [dataset_pair(train, test) for train, test in zip(Path(self.data_dir).glob("*_train"), Path(self.data_dir).glob("*_test"))]
	
	def tensorboard_logdir(self):
		return self.tbdir
	
	def new_session(self):
		return Path(self.models_dir) / "v1_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))