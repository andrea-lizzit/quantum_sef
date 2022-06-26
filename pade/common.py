import numpy as np

def jacobi_preconditioner(A):
	n = A.shape[0]
	dinv = np.zeros((n, n), dtype=A.dtype)
	for i in range(n):
		dinv[i, i] = 1.0 / A[i,i]
	return dinv