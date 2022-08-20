import logging
from typing import Union
import scipy.optimize
import numpy as np
import jax
import jax.numpy as jnp
from pade.common import jacobi_preconditioner, LossInfo

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

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
	""" Optimize params of a PadeModel using least GPU-accelerated gradient descent """

	logger.debug("Precise solution with GPU-accelerated minimization.")
	loss = jax.jit(pade_model_loss(z, s))
	gf = jax.grad(loss)
	x0 = complex_to_real(pparams)
	logger.debug(f"loss: {loss(x0)}")
	
	res = scipy.optimize.minimize(fun=loss, x0=x0, jac=gf,
					method="CG",
					callback=LossInfo(loss, gf, 10),
					options={"disp": True, "maxiter": precise_iter, "gtol": 1e-10})
	if not res.success:
		logger.info(f"optimization failed, reason: {res.message}")
	pparams = real_to_complex(res.x)
	return pparams