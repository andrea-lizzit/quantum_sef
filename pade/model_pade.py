from numbers import Complex, Real
from tkinter import N
from typing import Union
import numpy as np
import scipy
import scipy.optimize
import jax
import jax.numpy as jnp
from pade.common import jacobi_preconditioner

class LossInfo():
	def __init__(self, loss, jac=None, interval=1):
		self.loss = loss
		self.i = 0
		self.lastvalue = None
		self.interval = interval
		self.jac = jac
	def __call__(self, x):
		# print value if it is different from last
		self.i += 1
		value = self.loss(x)
		if self.lastvalue is None or self.lastvalue != value and self.i%self.interval == 0:
			if self.jac:
				print(f"{self.i}: {value}\tgradient norm: {np.linalg.norm(self.jac(x))}")
			else:
				print(f"{self.i}: {value}")
			self.lastvalue = value
		return False
		

def real_to_complex(x: Union[list, np.ndarray, jnp.ndarray]):
	# split real and imaginary parts
	return x[:len(x)//2] + 1j * x[len(x)//2:]

def pade_real_loss(A, b):
	def loss(x: Union[np.ndarray, jnp.ndarray]):
		x = real_to_complex(x)
		if isinstance(x, np.ndarray):
			return (np.mean(np.abs(np.dot(A, x) - b)**2))
		elif isinstance(x, jnp.ndarray):
			return (jnp.mean(jnp.abs(jnp.dot(A, x) - b)**2))
		else:
			return TypeError()
	return loss

def pade_real_loss_jax(A, b):
	def loss(arr: jnp.ndarray):
		x = real_to_complex(arr)
		return jnp.log(jnp.mean(jnp.abs(jnp.dot(A, x) - b)**2))
	return loss

def pade_model_loss(z, s):
	def loss(rparams):
		pparams = real_to_complex(rparams)
		a = pparams[0:len(pparams)//2]
		b = pparams[len(pparams)//2:]
		r = len(a)
		def scalar_loss(vz, vs):
			num = 0
			den = vz**r
			for i, ai in enumerate(a):
				num += ai*vz**i
			for i, bi in enumerate(b):
				den += bi*vz**i
			return num/den - vs
		vector_loss = jax.vmap(scalar_loss)
		return jnp.linalg.norm(vector_loss(z, s))
	return loss

def complex_to_real(x: Union[np.ndarray, jnp.ndarray]):
	# concatenate real and imaginary parts of x
	if isinstance(x, np.ndarray):
		return np.concatenate((np.real(x), np.imag(x)))
	elif isinstance(x, jnp.ndarray):
		return jnp.concatenate([jnp.real(x), jnp.imag(x)])
	else:
		return TypeError()

def lspparams(r, z, s):
	""" Least squares inversion of M*2r matrix that fits s """
	if (M := len(z)) != len(s):
		raise ValueError()
	K = np.zeros((M, 2*r), dtype=np.complex128)
	for i in range(M):
		for j in range(r):
			K[i, j] = z[i]**j
			K[i, r+j] = -s[i] * z[i]**j
	if np.isnan(K).any() or not np.isfinite(K).all():
		print(K)
		raise ValueError()
	y = jnp.array([z[i]**r * s[i] for i in range(M)], dtype=jnp.complex128)
	if np.isnan(y).any() or not np.isfinite(y).all():
		print(y)
		raise ValueError()
	pparams, residues, rank, sing = scipy.linalg.lstsq(K, y)
	condition = np.max(sing) / np.min(sing)
	if True or condition > 1e10:
		print("these are the singular values:\n", sing)
		print(f"Condition number of K before preconditioning: {condition}")
	# Pinv = jacobi_preconditioner(K)
	# K = np.dot(Pinv, K)
	# y =	np.dot(Pinv, y)
	# pparams, residues, rank, sing = scipy.linalg.lstsq(K, y)
	# condition = np.max(sing) / np.min(sing)
	# if True or condition > 1e10:
	# 	print("these are the singular values:\n", sing)
	# 	print(f"Condition number of K after preconditioning: {condition}")

	loss = jax.jit(pade_model_loss(z, s)) #pade_real_loss_jax(K, y)
	gf = jax.grad(loss)
	x0 = complex_to_real(pparams)
	print(f"loss: {loss(x0)}")
	# res = scipy.optimize.minimize(fun=loss, x0=x0, jac=gf,
	# 				method="CG",
	# 				callback=LossInfo(loss, gf, 10),
	# 				options={"gtol": 1e-35, "disp": True})
	# if not res.success:
	# 	print(f"optimization failed, reason: {res.message}")
	# pparams = real_to_complex(res.x)

	loss = np.sum(all_res := np.abs(y - np.dot(K, pparams))**2)
	if True or loss > 1:
		print(f"loss: {loss}")
		print("these are the residues:\n", all_res)
		# raise ValueError()

	return pparams

class PadeModel():
	def __init__(self, pparams):
		if len(pparams) % 2:
			raise ValueError()
		self.pparams = pparams
	def approx_num_den(self, z: Complex) -> Complex:
		a = self.pparams[0:len(self.pparams)//2]
		b = self.pparams[len(self.pparams)//2:]
		r = len(a)
		num = 0
		with np.errstate(over='raise'):
			try:
				den = z**r
			except FloatingPointError as e:
				print(f"{z}**{r}")
				raise e
		for i, ai in enumerate(a):
			num += ai*z**i
		for i, bi in enumerate(b):
			den += bi*z**i
		return num, den
	def __call__(self, z: Complex) -> Complex:
		num, den = self.approx_num_den(z)		
		return num / den
	def rho(self, w: Real, delta: Real = 0.01):
		""" evaluate spectral function, delta Rydberg over the real line """
		return -1/np.pi * np.imag(w + delta*1j)
	def physical(self, w, tol=0):
		for wi in w:
			if self.rho(wi) < tol:
				return False
		return True
		
class AverageModel(PadeModel):
	def __init__(self, models, weights):
		self.models = models
		self.weights = weights
	def __call__(self, z: Complex) -> Complex:
		res = 0
		for model, weight in zip(self.models, self.weights):
			res += model(z) * weight
		res /= np.sum(self.weights)
		return res

def model_pade(z, s, M=range(50, 99, 4), N=range(50, 99, 4), n0=[1], plot=True):
	z, s, M, N = np.array(z), np.array(s), np.array(M), np.array(N)
	models = []
	for m in M:
		if m%2:
			pass #raise ValueError()
		for n in N:
			if n%2:
				raise ValueError()
			if n > m:
				continue
			samples_i = np.linspace(len(z)*0., 0.25*len(z)-1, num=m, dtype=np.int)
			r = n // 2
			pparams = lspparams(r, z[samples_i], s[samples_i])
			models.append(model := PadeModel(pparams))
			if plot:
				# plot real and imaginary parts
				import matplotlib.pyplot as plt
				indices = np.where(np.abs(model(z)) < 1e3)[0]
				plt.plot(np.imag(z[indices]), np.real(model(z[indices])), label=f"m={m}, n={n}", color="#c33")
				plt.plot(np.imag(z[indices]), np.imag(model(z[indices])), label=f"m={m}, n={n}", color="#933")
				# likewise with s
				plt.plot(np.imag(z), np.real(s), label=f"m={m}, n={n}", color="#3c3")
				plt.plot(np.imag(z), np.imag(s), label=f"m={m}, n={n}", color="#393")
				# plot points of s corresponding to samples_i with red circles
				plt.plot(np.imag(z[samples_i]), np.real(s[samples_i]), "ro", label=f"m={m}, n={n}")
				plt.plot(np.imag(z[samples_i]), np.imag(s[samples_i]), "ro", label=f"m={m}, n={n}")
				plt.legend()
				plt.show()

	# average diagonal
	w = np.imag(z)
	weights = [1 if model.physical(w) else 0.001 for model in models]
	avgdiagmodel = AverageModel(models, weights)
	return avgdiagmodel