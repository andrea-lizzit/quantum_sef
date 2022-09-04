import numpy as np
import torch
import itertools
from tqdm import tqdm, trange
from neural.xydataset import XYDataset
import multipole.common as mp

device = torch.device("cpu")


class SpectralFunction():
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
		return SpectralFunction(peaks, heights, widths)

	@property
	def w(self):
		return np.linspace(-20, 20, num=481)
	
	@property
	def iw(self):
		return np.linspace(0, 20, num=241)

def spectraldataset(mult=1000):
	set1 = itertools.islice(SpectralGen(1), 18 * mult)
	set2 = itertools.islice(SpectralGen(2), 18 * mult)
	set3 = itertools.islice(SpectralGen(3), 20 * mult)
	set4 = itertools.islice(SpectralGen(4), 10 * mult)
	set5 = itertools.islice(SpectralGen(5), 10 * mult)
	set6 = itertools.islice(SpectralGen(6), 10 * mult)
	set7 = itertools.islice(SpectralGen(7), 10 * mult)
	set8 = itertools.islice(SpectralGen(8), 10 * mult)
	spectralgens = itertools.chain(set1, set2, set3, set4, set5, set6, set7, set8)
	xdata = list()
	ydata = list()
	for gen in tqdm(spectralgens, total=106 * mult):
		x = gen.imag()
		y = gen.real()
		xdata.append(torch.tensor(np.stack([np.real(x), np.imag(x)]), dtype=torch.float32, device=device))
		ydata.append(torch.tensor(y, dtype=torch.float32, device=device))
	return XYDataset(xdata, ydata)

rng = np.random.default_rng()

def se_dataset(mult=1000):
	xdata, ydata = list(), list()
	w = np.linspace(-20, 20, num=481, dtype=np.complex64)
	iw = np.linspace(0, 20, num=241)

	for n in tqdm([4]):
		for i in trange(mult*10):
			# choose n poles at random and build the MPParams
			poles = list()
			for j in range(n):
				a = complex(*rng.standard_normal(2))*(0.4-0.02) + 0.02
				b = complex(rng.random()*5+1, rng.random() * 0.37 + 0.05)
				if j%2:
					b *= -1
				poles.append(mp.Pole(a, b))
			bias = rng.random() * 4
			params = mp.MPParams(bias, poles)
			# sample the function on re and im axes
			x = mp.multipole(iw*1j, params)
			y = mp.multipole(w, params)
			xdata.append(torch.tensor(np.stack([np.real(x), np.imag(x)]), dtype=torch.float32, device=device))
			ydata.append(torch.tensor(np.stack([np.real(y), np.imag(y)]), dtype=torch.float32, device=device))
	return XYDataset(xdata, ydata)