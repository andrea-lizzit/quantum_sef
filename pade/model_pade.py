import logging
from numbers import Complex, Real
from tkinter import N
from typing import Union
import numpy as np
import scipy
import scipy.optimize
import jax
import jax.numpy as jnp
from pade.common import jacobi_preconditioner
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class LossInfo():
	def __init__(self, loss, jac=None, interval=1):
		self.loss = loss
		self.i = 0
		self.lastvalue = None
		self.interval = interval
		self.jac = jac
	def __call__(self, x):
		""" Log value if it is different from last or an interbal has passed """
		self.i += 1
		value = self.loss(x)
		if self.lastvalue is None or self.lastvalue != value and self.i%self.interval == 0:
			if self.jac:
				logger.debug(f"{self.i}: {value}\tgradient norm: {np.linalg.norm(self.jac(x))}")
			else:
				logger.debug(f"{self.i}: {value}")
			self.lastvalue = value
		return False
		

def real_to_complex(x: Union[list, np.ndarray, jnp.ndarray]):
	""" Split real and imaginary parts of x """
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
	""" Loss function as MSE with respect to s """
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
		return jnp.mean(jnp.abs(vector_loss(z, s))**2)
	return loss

def complex_to_real(x: Union[np.ndarray, jnp.ndarray]):
	""" Concatenate real and imaginary parts of x """
	if isinstance(x, np.ndarray):
		return np.concatenate((np.real(x), np.imag(x)))
	elif isinstance(x, jnp.ndarray):
		return jnp.concatenate([jnp.real(x), jnp.imag(x)])
	else:
		return TypeError()

def lspparams(r, z, s, precondition=False):
	""" Least squares inversion of M*2r matrix that fits s """
	if (M := len(z)) != len(s):
		raise ValueError()
	K = np.zeros((M, 2*r), dtype=np.complex128)
	for i in range(M):
		for j in range(r):
			K[i, j] = z[i]**j
			K[i, r+j] = -s[i] * z[i]**j
	if np.isnan(K).any() or not np.isfinite(K).all():
		logger.error("K is not finite or contains NaN")
		logger.error(K)
		raise ValueError()
	y = jnp.array([z[i]**r * s[i] for i in range(M)], dtype=jnp.complex128)
	if np.isnan(y).any() or not np.isfinite(y).all():
		logger.error("y is not finite or contains NaN")
		raise ValueError()
	pparams, residues, rank, sing = scipy.linalg.lstsq(K, y)
	condition = np.max(sing) / np.min(sing)
	if condition > 1e20:
		logger.debug(f"Condition number of K (without preconditioning) is very high: {condition}")
		logger.debug(f"these are the singular values:\n{sing}")
	if precondition:
		logger.warn("Jacobi preconditioning is enabled. This is dangerous as in test cases it increased the condition number. This kind of matrix is not a good target for Jacobi preconditioning.")
		Pinv = jacobi_preconditioner(K)
		K = np.dot(Pinv, K)
		y =	np.dot(Pinv, y)
		pparams, residues, rank, sing = scipy.linalg.lstsq(K, y)
		condition = np.max(sing) / np.min(sing)
		if condition > 1e16:
			logger.debug(f"Condition number of K (after preconditioning) is very high: {condition}")
			logger.debug(f"these are the singular values:\n{sing}")

	if (loss_val := pade_model_loss(z, s)(pparams)) > 1:
		logger.info(f"loss is high: {loss_val}")
		all_res = np.abs(y - np.dot(K, pparams))**2
		# logger.debug(f"these are the residues:\n{all_res}")
	return pparams

def optimize_pparams(pparams, z, s, precise_iter=0):
	logger.debug("Precise solution with GPU-accelerated minimization.")
	loss = jax.jit(pade_model_loss(z, s))
	gf = jax.grad(loss)
	x0 = complex_to_real(pparams)
	logger.debug(f"loss: {loss(x0)}")
	
	res = scipy.optimize.minimize(fun=loss, x0=x0, jac=gf,
					method="CG",
					callback=LossInfo(loss, gf, 10),
					options={"disp": True, "maxiter": precise_iter, "gtol": 1e-6})
	if not res.success:
		logger.info(f"optimization failed, reason: {res.message}")
	pparams = real_to_complex(res.x)
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
		return -1/np.pi * np.imag(self.__call__(w + delta*1j))
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


class AverageLSModel(AverageModel):
	def __init__(self, models, w):
		self.models = models
		self.weights = [model.physical(w) for model in models]

class AverageSimilarModel(AverageModel):
	def __init__(self, models, w):
		self.models = models
		self.w = w
		self.weights = [self.Dc(i) for i in range(len(self.models))]
		if not any(self.weights):
			raise RuntimeError("No physical solutions")
	def __call__(self, z: Complex) -> Complex:
		res = 0
		for model, weight in zip(self.models, self.weights):
			res += model(z) * weight
		res /= np.sum(self.weights)
		return res
	def Dc(self, i):
		deviation = 0
		for j, model in enumerate(self.models):
			if j == i:
				continue
			if not model.physical(self.w):
				continue
			deviation += np.sum(np.abs(model.rho(self.w) - self.models[i].rho(self.w)))
		return deviation

def model_pade(z, s, M=range(50, 99, 4), N=range(50, 99, 4), n0=[1], plot=False, precise_iter=0, modeltype='avgLS', **kwargs):
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
			samples_i = np.linspace(0, len(z)-1, num=m, dtype=np.int)
			r = n // 2
			pparams = lspparams(r, z[samples_i], s[samples_i], **kwargs)
			if precise_iter:
				pparams = optimize_pparams(pparams, z, s, precise_iter)
			models.append(model := PadeModel(pparams))
			if plot:
				fig, ax = plt.subplots(1, 2, figsize=(10, 5))
				ax[0].plot(np.imag(z), np.real(model(z)), label=f"m={m}, n={n}", color="#c33")
				ax[0].plot(np.imag(z), np.imag(model(z)), label=f"m={m}, n={n}", color="#933")
				ax[0].legend()
				ax[0].plot(np.imag(z), np.real(s), label=f"m={m}, n={n}", color="#3c3")
				ax[0].plot(np.imag(z), np.imag(s), label=f"m={m}, n={n}", color="#393")
				# # plot points of s corresponding to samples_i with red circles
				ax[0].plot(np.imag(z[samples_i]), np.real(s[samples_i]), "ro", label=f"m={m}, n={n}", ms=2)
				ax[0].plot(np.imag(z[samples_i]), np.imag(s[samples_i]), "ro", label=f"m={m}, n={n}", ms=2)
				ax[0].legend()
				# get indexes where model is smaller than 1e4
				# plot like above but only points smaller than 1e4. Evaluate model, get indices, plot
				y = model(-z*1j)
				idx = np.where(np.abs(y) < 1e4)
				ax[1].plot(np.imag(z[idx]), np.real(y[idx]), label=f"m={m}, n={n}", color="#c33", ms=2)
				ax[1].plot(np.imag(z[idx]), np.imag(y[idx]), label=f"m={m}, n={n}", color="#393", ms=2)
				ax[1].plot(np.imag(z[idx]), model.rho(np.imag(z[idx])), label=f"m={m}, n={n}", color="#3c3", ms=2)
				ax[1].legend()
				plt.show()
	# average diagonal
	w = np.imag(z)
	weights = [1 if model.physical(w) else 0.001 for model in models]
	print(f"there are {np.sum([model.physical(w) for model in models])} models with physical properties")
	if modeltype == 'avgLS':
		return AverageLSModel(models, w)
	else:
		avgdiagmodel = AverageSimilarModel(models, w)
	return avgdiagmodel