import numpy as np
import torch
from torch.utils.data import Dataset
import itertools
from tqdm import tqdm
from pathlib import Path

device = torch.device("cpu") # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SpectralSelfEnergy():
	def __init__(self, peaks, heights, widths):
		self.peaks = peaks
		self.heights = heights
		self.widths = widths

	def real(self):
		w = np.linspace(-20, 20, num=481)
		gaussians = [np.exp(-((w - peak) / width) ** 2) * height for peak, width, height in zip(self.peaks, self.widths, self.heights)]
		return np.sum(gaussians, axis=0)

	def imag(self):
		z = [complex(0, x) for x in np.linspace(0, 20, num=241)]
		z[0] += z[1]/2
		res = [self(z) for z in z]
		return res

	def __call__(self, z) -> complex:
		rho = self.real()
		w = np.linspace(-20, 20, num=481)
		return np.sum(rho / (z - w))


class SpectralGen:
	""" An iterator that returns a spectral function with n peaks """
	def __init__(self, npeaks):
		self.npeaks = npeaks

	def __iter__(self):
		return self

	def __next__(self):
		# choose n_peaks random peaks
		peaks = np.random.rand(self.npeaks) * 30 - 20
		# for each peak, pick a random width and height
		heights = np.random.rand(self.npeaks) * 0.8 + 0.6
		widths = np.random.rand(self.npeaks) * 0.9 + 0.75
		# for each peak, create a gaussian
		area = np.sum([np.sqrt(np.pi * width) for width in widths])
		heights = [height / area for height in heights]
		# add them together
		return SpectralSelfEnergy(peaks, heights, widths)

	@property
	def w(self):
		return np.linspace(-20, 20, num=481)
	
	@property
	def iw(self):
		return np.linspace(0, 20, num=241)

class SpectralDataset(Dataset):
	def __init__(self, mult=1000):
		set1 = itertools.islice(SpectralGen(1), 18 * mult)
		set2 = itertools.islice(SpectralGen(1), 18 * mult)
		set3 = itertools.islice(SpectralGen(1), 20 * mult)
		set4 = itertools.islice(SpectralGen(1), 10 * mult)
		set5 = itertools.islice(SpectralGen(1), 10 * mult)
		set6 = itertools.islice(SpectralGen(1), 10 * mult)
		set7 = itertools.islice(SpectralGen(1), 10 * mult)
		set8 = itertools.islice(SpectralGen(1), 10 * mult)
		self.spectralgens = itertools.chain(set1, set2, set3, set4, set5, set6, set7, set8)
		self.xdata = list()
		self.ydata = list()
		for gen in tqdm(self.spectralgens, total=106 * mult):
			x = gen.imag()
			y = gen.real()
			self.xdata.append(torch.tensor(np.stack([np.real(x), np.imag(x)]), dtype=torch.float32, device=device))
			self.ydata.append(torch.tensor(y, dtype=torch.float32, device=device))

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

class StorageSpectralDataset(Dataset):
	def __init__(self, directory):
		directory = Path(directory)
		self.xdata = torch.load(str(directory / "xdata.pt"))
		self.ydata = torch.load(str(directory / "ydata.pt"))

	def __len__(self):
		return len(self.xdata)

	def __getitem__(self, idx):
		return self.xdata[idx], self.ydata[idx]

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	spectralgen = SpectralGen(npeaks=4)
	selfenergy = next(spectralgen)

	# create two subplots
	fig, ax = plt.subplots(2, 1)
	# plot selfenergy.real() on the first subplot
	ax[0].plot(spectralgen.w, selfenergy.real())
	# plot real and imaginary parts of selfenergy.imag() on the second subplot
	ax[1].plot(spectralgen.iw, np.real(selfenergy.imag()))
	ax[1].plot(spectralgen.iw, np.imag(selfenergy.imag()))

	plt.show()