from collections import namedtuple


Pole = namedtuple("Pole", ["a", "b"])
MPParams = namedtuple("MPParams", ["bias", "poles"])

def multipole(z, params):
	"""Multipole function to fit."""
	v = params.bias
	for pole in params.poles:
		v += pole.a / (z - pole.b)
	return v

def model_mpparams(params):
	return lambda z: multipole(z, params)