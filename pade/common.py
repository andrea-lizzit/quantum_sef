import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def jacobi_preconditioner(A):
	n = A.shape[0]
	dinv = np.zeros((n, n), dtype=A.dtype)
	for i in range(n):
		dinv[i, i] = 1.0 / A[i,i]
	return dinv

class LossInfo():
	def __init__(self, loss, jac=None, interval=1):
		self.loss = loss
		self.i = 0
		self.lastvalue = None
		self.interval = interval
		self.jac = jac
	def __call__(self, x):
		""" Log value if it is different from last or an interval has passed """
		self.i += 1
		value = self.loss(x)
		if self.lastvalue is None or self.lastvalue != value and self.i%self.interval == 0:
			if self.jac:
				logger.debug(f"{self.i}: {value}\tgradient norm: {np.linalg.norm(self.jac(x))}")
			else:
				logger.debug(f"{self.i}: {value}")
			self.lastvalue = value
		return False