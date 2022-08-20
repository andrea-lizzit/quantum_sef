from turtle import color
import numpy as np
from dyson import GW
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm

results = namedtuple("results", ["res", "summary", "plot"])
class Variator():
	def __init__(self, params):
		self.params = params
	def __call__(self):
		return NotImplementedError()
	def generate(self, n):
		""" Generator that yields n sets of parameters """
		for i in range(n):
			yield self()

class GaussianVariator(Variator):
	def __init__(self, params, sigma=0.01):
		super().__init__(params)
		self.sigma = sigma
	def __call__(self):
		""" Returns a set of parameters that differ from the originals by gaussian noise """
		return self.params * np.random.normal(1, self.sigma, size=self.params.shape)
	def __repr__(self):
		return "GaussianVariator(%s, %s)" % (1, self.sigma)

def sensitivity(Model, params, z, fit_model, Ei, offset):
	""" @model: a model to be tested
		@z: points on the imaginary axis
		@fit_model: a function that fits a model to the data """
	gen = GaussianVariator(params)

	res = []
	n_samples = 500
	for params in tqdm(gen.generate(n_samples), total=n_samples):
		model = Model(params)
		s = model(z)
		fmodel = fit_model(model, z, s)
		res.append(GW(Ei, fmodel, offset))

		# debug: show plots
		# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
		# model.plot(ax, z, s)
		# plt.show()
	res_orig = GW(Ei, Model(params), offset)

	summary = "Sensitivity to parameter variation\n" \
			+ "Model: %s\n" % Model.__name__ \
			+ "Variator: %s\n" % repr(gen) \
			+ "Parameter dimensions: %s\n" % params.shape \
			+ "Number of samples: %s\n" % n_samples \
			+ f"Output: {np.real(np.mean(res)):.5} +- {np.std(res):.3}\n" \
			+ f"Original: {np.real(res_orig):.5}\n" \
			+ f"Difference: {np.real(np.mean(res) - res_orig):.5}"
	fig, ax = plt.subplots()
	ax.hist(np.real(res))
	ax.axvline(x=np.real(res_orig), color="g", label="original")
	ax.set_xlabel("E (eV)")
	ax.set_ylabel("counts")

	return results(res, summary, fig)