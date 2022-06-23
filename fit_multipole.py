#!/usr/bin/env python

import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize
from collections import namedtuple
from dyson import dyson
from qe_utils import load_qe_se


Pole = namedtuple("Pole", ["a", "b"])
MPParams = namedtuple("MPParams", ["bias", "poles"])

def multipole(z, params):
	"""Multipole function to fit."""
	v = params.bias
	for pole in params.poles:
		v += pole.a / (z - pole.b)
	return v

def make_mpparams(x):
	""" Get a numpy array of reals and return an MPParams object """
	if len(x) % 4 != 2:
		raise ValueError()
	n_poles = (len(x)-1)//4
	poles = []
	for i in range(n_poles):
		a = x[2+4*i] + x[3+4*i]*1j
		b = x[4+4*i] + x[5+4*i]*1j
		poles.append(Pole(a=a, b=b))
	params = MPParams(bias=x[0] + x[1]*1j, poles=poles)
	return params

def unmake_params(params):
	""" Get an MPParams object and return a numpy array of reals """
	x = [np.real(params.bias), np.imag(params.bias)]
	for pole in params.poles:
		x.append(np.real(pole.a))
		x.append(np.imag(pole.a))
		x.append(np.real(pole.b))
		x.append(np.imag(pole.b))
	return np.array(x)

def chi2(z, s, f):
	res = np.array(list(map(lambda vz, vs: f(vz) - vs, z, s)))
	chi2 = np.sum(np.absolute(res)**2)
	return chi2

def chi2_jax(z, s, f):
	resf = lambda vz, vs: jnp.abs(f(vz) - vs)**2
	vf = jax.vmap(resf)
	res = vf(z, s)
	return res.sum()


def fit_multipole_jax(z, s, n_poles):
	# parameter initialization (taken from quantum espresso)
	poles = []
	for i in range(1, n_poles+1):
		a = complex(i, 0)
		b = complex(i*0.5 * (-1)**i, -0.01)
		poles.append(Pole(a=a, b=b))

	params = MPParams(bias = 0, poles = poles)
	x0 = unmake_params(params)
	
	def loss(x):
		params = make_mpparams(x)
		return chi2_jax(z, s, lambda vz: multipole(vz, params))

	gf = jax.grad(loss)
	print(f"chi iniziale {loss(x0)} \ngrad iniziale {gf(x0)}")
	res = scipy.optimize.minimize(fun=loss, x0=x0, jac=gf, method="BFGS")
	print(f"chi finale {loss(res.x)} \ngrad finale {gf(res.x)}")
	if not res.success:
		print(f"optimization failed, reason: {res.message}")
	return make_mpparams(res.x), res.fun


def fit_multipole(z, s, n_poles, delta=0.1, iterations=10000):
	"""Fit a multipole function."""


	def gradient(z, params):
		"""Gradient of the multipole function"""
		# gradient of bias
		bias = sum(map(lambda vz, vs: multipole(vz, params) - vs, z, s)) # conj?
		grad = MPParams(bias=bias, poles=[])
		norm = abs(bias)**2

		for pole in params.poles:
			# gradient a
			gf = lambda vz, vs: (multipole(vz, params) - vs) / np.conjugate(vz - pole.b)
			ga = sum(map(gf, z, s))
			#ga = np.conjugate(ga) # coniugare?
			norm += abs(ga)**2
			# gradient b
			gf = lambda vz, vs: (multipole(vz, params) - vs) * np.conjugate(pole.a / (vz - pole.b)**2)
			gb = sum(map(gf, z, s))
			#gb = -np.conjugate(gb) # coniugare?
			norm += abs(gb)**2
			
			grad.poles.append(Pole(a=ga, b=gb))

		# normalize
		ngrad = MPParams(bias = bias / norm, poles=[])
		for pole in grad.poles:
			ngrad.poles.append(Pole(a=pole.a/norm, b=pole.b/norm))
		
		return ngrad

	# parameter initialization (taken from quantum espresso)
	poles = []
	for i in range(1, n_poles+1):
		a = complex(i, 0)
		b = complex(i*0.5 * (-1)**i, -0.01)
		poles.append(Pole(a=a, b=b))

	params = MPParams(bias = 0, poles = poles)
	vchi2 = chi2(z, s, lambda vz: multipole(vz, params))

	for i in range(iterations):
		# one step of minimization
		grad = gradient(z, params)
		
		params = params._replace(bias = params.bias - delta * grad.bias)
		for j in range(len(params.poles)):
			a = params.poles[j].a - delta * grad.poles[j].a
			b = params.poles[j].b - delta * grad.poles[j].b
			params.poles[j] = params.poles[j]._replace(a=a, b=b)

		new_vchi2 = chi2(z, s, lambda vz: multipole(vz, params))
		
		if new_vchi2 > vchi2:
			delta *= 0.5
			print(f"lowering step size to {delta}")
		vchi2 = new_vchi2
		
	return params, vchi2

if __name__ == "__main__":
	import csv, argparse, yaml
	import matplotlib.pyplot as plt
	parser = argparse.ArgumentParser()
	parser.add_argument("file_real")
	parser.add_argument("file_im")
	parser.add_argument("--HF_energies", help="yaml file containing Hartree-Fock energies of HOMO, LUMO, and target")
	parser.add_argument("--n_poles", type=int, default=4)
	parser.add_argument("--n_fit", type=int, default=50)
	parser.add_argument("--iterations", type=int, default=1000)
	args = parser.parse_args()

	# read files up to row n_fit
	t = load_qe_se(args.file_real, args.file_im, args.n_fit, positive=True)
	z, s = jnp.array(t['z']), jnp.array(t['s'])

	#params, vchi2 = fit_multipole(z, s, args.n_poles, iterations=args.iterations)
	params, vchi2 = fit_multipole_jax(z, s, args.n_poles)

	print(f"chi2: {vchi2}")
	print(f"a_0: {params.bias}")
	for pole in params.poles:
		print(f"a: {pole.a}\tb: {pole.b}")
	
	fit_s = [multipole(vz, params) for vz in z]
	z, s, fit_s = np.array(z), np.array(s), np.array(fit_s)
	plt.plot(np.imag(z), np.real(fit_s), label="fit s real")
	plt.plot(np.imag(z), np.real(s), label="reference s real")
	plt.plot(np.imag(z), np.imag(fit_s), label="fit s imag")
	plt.plot(np.imag(z), np.imag(s), label="reference s imag")
	plt.legend()
	plt.show()

	if args.HF_energies:
		with open(args.HF_energies) as fd:
			energies = yaml.safe_load(fd)
			e_HOMO, e_LUMO, e_target = energies["HOMO"], energies["LUMO"], energies["target"]
		
		offset = (e_LUMO + e_HOMO) / 2
		E0 = e_target - offset
		E = dyson(E0, lambda w: multipole(w, params))
		E += offset
		print(f"GW energy: {E}")
