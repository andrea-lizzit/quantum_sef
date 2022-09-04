from datetime import datetime
from pathlib import Path
from collections import namedtuple
from functools import cache

dataset_pair = namedtuple("dataset_pair", ["train", "test"])

class StorageManager():
	""" Consistent management of directories and locations for neural models """

	def __init__(self,
					models_dir = "out/models",
					data_dir = "out/datasets",
					tensorboard_logdir="out/runs",
					version=1):
		self.models_dir = Path(models_dir)
		self.data_dir = Path(data_dir)
		self.tbdir = Path(tensorboard_logdir)
		self.version = version

	def sessions(self):
		""" Returns a list of all sessions in the models directory """
		return list(self.models_dir.glob(f"v{self.version}_*"))
	
	def models(self, session):
		""" Returns a list of all models in the session """
		return list(Path(session).glob("*.pt"))
	
	def last_model(self):
		""" Returns the path to the last model in the models directory """
		return self.models(self.sessions()[-1])[-1]
	
	def datasets(self):
		""" Returns a list of all datasets in the data directory """
		return [dataset_pair(dir / "train", dir/"test") for dir in Path(self.data_dir).glob(f"v{self.version}_dataset_*")]
	
	def tensorboard_logdir(self):
		return self.tbdir

	@cache	
	def new_session(self):
		return self.models_dir / "v{}_{}".format(self.version, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
	
	@cache
	def new_dataset(self):
		return self.data_dir / "v{}_dataset_{}".format(self.version, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))