#!/usr/bin/env python

import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize
from multipole.common import Pole, MPParams, model_mpparams


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

def unmake_mpparams(params):
	""" Get an MPParams object and return a numpy array of reals """
	x = [np.real(params.bias), np.imag(params.bias)]
	for pole in params.poles:
		x.append(np.real(pole.a))
		x.append(np.imag(pole.a))
		x.append(np.real(pole.b))
		x.append(np.imag(pole.b))
	return np.array(x)

def chi2(z, s, f):
	resf = lambda vz, vs: jnp.abs(f(vz) - vs)**2
	vf = jax.vmap(resf)
	res = vf(z, s)
	return res.sum()

def complex_chi2(a, b):
	diff = jnp.abs(jnp.subtract(a, b))
	return jnp.multiply(diff, diff).sum()


def fit_mpparams(z, s, n_poles):
	# parameter initialization (taken from quantum espresso)
	poles = []
	for i in range(1, n_poles+1):
		a = complex(i, 0)
		b = complex(i*0.5 * (-1)**i, -0.01)
		poles.append(Pole(a=a, b=b))

	params = MPParams(bias = 0, poles = poles)
	x0 = unmake_mpparams(params)
	
	def loss(x):
		params = make_mpparams(x)
		model = model_mpparams(params)
		vmodel = jax.vmap(model)
		y = vmodel(z)
		return complex_chi2(y, s)

	gf = jax.grad(loss)
	print(f"chi iniziale {loss(x0)} \ngrad iniziale {gf(x0)}")
	res = scipy.optimize.minimize(fun=loss, x0=x0, jac=gf, method="BFGS")
	print(f"chi finale {loss(res.x)} \ngrad finale {gf(res.x)}")
	#res = scipy.optimize.minimize(fun=loss, x0=res.x, jac=gf, method="Nelder-Mead", options={'disp': True})
	#print(f"chi finale {loss(res.x)} \ngrad finale {gf(res.x)}")

	if not res.success:
		print(f"optimization failed, reason: {res.message}")
	return make_mpparams(res.x), res.fun

def model_multipole(z, s, n_poles):
	params = fit_mpparams(z, s, n_poles)[0]
	return model_mpparams(params)