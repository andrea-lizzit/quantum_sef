#!/usr/bin/env python

import numpy as np
import jax
import jax.numpy as jnp
from qe_utils import load_qe_se
from multipole.model_multipole import model_multipole
from multipole.model_qe import model_qe

def fit(method, *args, **kwargs):
	if method == "qe":
		return model_qe(*args, **kwargs)
	if method == "multipole":
		return model_multipole(*args, **kwargs)
	return None

if __name__ == "__main__":
	import argparse
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

	model = model_multipole(z, s, args.n_poles)

	fit_s = [model(vz) for vz in z]
	z, s, fit_s = np.array(z), np.array(s), np.array(fit_s)
	plt.plot(np.imag(z), np.real(fit_s), label="fit s real")
	plt.plot(np.imag(z), np.real(s), label="reference s real")
	plt.plot(np.imag(z), np.imag(fit_s), label="fit s imag")
	plt.plot(np.imag(z), np.imag(s), label="reference s imag")
	plt.legend()
	plt.show()

	# if args.HF_energies:
	# 	with open(args.HF_energies) as fd:
	# 		energies = yaml.safe_load(fd)
	# 		e_HOMO, e_LUMO, e_target = energies["HOMO"], energies["LUMO"], energies["target"]
		
	# 	offset = (e_LUMO + e_HOMO) / 2
	# 	E0 = e_target - offset
	# 	E = dyson(E0, lambda w: multipole(w, params))
	# 	E += offset
	# 	print(f"GW energy: {E}")
