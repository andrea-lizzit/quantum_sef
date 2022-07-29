import numpy as np


class Variator():
	def __init__(self, params):
		self.params = params
	def __call__(self):
		""" Returns a set of parameters that differ from the originals by gaussian noise """
		return self.params * np.random.normal(1, 0.1, size=self.params.shape)
	def generate(self, n):
		""" Generator that yields n sets of parameters """
		for i in range(n):
			yield self()

gen = Variator(model.params)

res = []
for params in gen.generate(10):
	model = Model(params)
	se = model(z)
	fmodel = Model.fit(se)
	params - fmodel.params
	res.append(params - fmodel.params)

print(res)