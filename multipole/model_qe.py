import numpy as np
import qe_utils
from multipole.common import multipole

def model_qe(gww_out, orbital):
	params = qe_utils.load_gww_fit(gww_out)[orbital-1]
	def inner(z):
		if np.real(z) > 0:
			return multipole(z, params)
		else:
			sigma = multipole(np.conj(z), params) # like in qe
			return np.conj(sigma)
	return inner