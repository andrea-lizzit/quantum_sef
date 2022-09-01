import torch
from pathlib import Path


class XYDataset():
	def __init__(self, xdata, ydata):
		self.xdata = xdata
		self.ydata = ydata

	def __len__(self):
		return len(self.xdata)

	def __getitem__(self, idx):
		return self.xdata[idx], self.ydata[idx]

	def save(self, directory):
		directory = Path(directory)
		if not directory.exists():
			directory.mkdir()
		torch.save(torch.stack(self.xdata, dim=0), str(directory / "xdata.pt"))
		torch.save(torch.stack(self.ydata, dim=0), str(directory / "ydata.pt"))

	@classmethod
	def load(cls, directory):
		directory = Path(directory)
		xdata = torch.load(str(directory / "xdata.pt"))
		ydata = torch.load(str(directory / "ydata.pt"))
		return cls(xdata, ydata)
