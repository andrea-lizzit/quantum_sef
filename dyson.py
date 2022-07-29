#!/usr/bin/env python

import numpy as np
from qe_utils import load_gww_energies, load_gww_fit, load_qe_se
import fit
import logging
from pade.model_pade import PadeModel

RY = 13.605693122994

def multipole(z, params):
	"""Multipole function to fit."""
	v = params.bias
	for pole in params.poles:
		v += pole.a / (z - pole.b)
	return v

def dyson(E0, correlation, precision=0.000001):
	E = E0
	for _ in range(10):
		print(f"new E: {E}")
		prev_E = E
		E = E0 + correlation(E)
		print(f"self energy {correlation(E)}")
		if abs(prev_E - E) < precision * abs(E0):
			pass
		if abs(prev_E - E) > 10*abs(E0):
			raise RuntimeError("self-consistent calculation diverges")
	return E

def dyson_s(E0, e0, correlation, precision=0.000001):
	E = e0
	for _ in range(10):
#		print(f"new E: {E}")
		prev_E = E
		E = E0 + correlation(E)
		if abs(prev_E - E) < precision * abs(E0):
			pass
		if abs(E) > 10*abs(E0):
			raise RuntimeError("self-consistent calculation diverges")
	return E

def GW(qe_E, correlation):
	E0 = qe_E["DFT"]/RY - offset
	E0 = complex(E0, 0)
	E0 = qe_E["HF-pert"]/RY - offset + correlation(E0)

	E = dyson_s(qe_E["HF-pert"] / RY - offset, E0, correlation)
	E += offset
	E *= RY
	return E


if __name__ == "__main__":
	import csv, argparse, yaml
	import matplotlib.pyplot as plt
	from jax.config import config
	config.update("jax_enable_x64", True)

	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	fh = logging.FileHandler("quantum-sef.log")
	fh.setLevel(logging.INFO)
	formatter = logging.Formatter('%(name)s %(levelname)s %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	parser = argparse.ArgumentParser()
	parser.add_argument("--orbital", type=int)
	subparsers = parser.add_subparsers(dest="subparser")
	parser.add_argument("gww_out", help="filename of the gww output")
	p_loadparams = subparsers.add_parser("gwwparams")
	p_loadparams.add_argument("--self_energy", "-s", help="qe self-energy file for comparison")
	p_multipole = subparsers.add_parser("multipole")
	p_multipole.add_argument("prefix")
	p_multipole.add_argument("n_poles", type=int)
	p_pade = subparsers.add_parser("pade")
	p_pade.add_argument("prefix")
	p_pade.add_argument("--type", choices=["avgLS", "avgsimilar"])
	p_a = subparsers.add_parser("gwwfit")
	p_a.add_argument("self_energy", help="prefix and suffix of name of the file containing the self-energy values")
	p_a.add_argument("gww_out", help="filename of the gww output")
	p_a.add_argument("--qe", action="store_true", help="use qe self-energy")
	args = parser.parse_args()
	
	E = load_gww_energies(args.gww_out)
	if args.orbital:
		orbitals = [args.orbital]
	else:
		orbitals = [1, 2, 3, 4, 5]

	offset = (E[5]["DFT"] + 0) / (2 * RY)
	print(f'offset {offset} = {E[5]["DFT"]} + {0} / 2RY')

	if args.subparser == "gwwparams":
		get_model = lambda orbital: fit.fit("qe", args.gww_out, orbital)
	if args.subparser == "multipole":
		def get_model(orbital):
			filename_real = args.prefix + "-re_on_im0000" + str(orbital)
			filename_imag = args.prefix + "-im_on_im0000" + str(orbital)
			qe_data = load_qe_se(filename_real, filename_imag, positive=True)
			z, s = qe_data["z"], qe_data["s"]
			return fit.fit("multipole", z, s, args.n_poles)
	if args.subparser == "pade":
		def get_model(orbital):
			filename_real = args.prefix + "-re_on_im0000" + str(orbital)
			filename_imag = args.prefix + "-im_on_im0000" + str(orbital)
			qe_data = load_qe_se(filename_real, filename_imag, positive=True)
			z, s = qe_data["z"], qe_data["s"]
			#return PadeModel.from_specs(z, s, 8, 8)
			return fit.fit("pade", z, s, precise_iter=0, M=range(8, 80, 8), N=[8, 80, 8], modeltype=args.type)
	# orbitals = load_gww_energies(args.gww_out).keys()
	for i in orbitals:
		E[i]["GWC"] = GW(E[i], get_model(i))
		# print(f"state {i}; GW energy: {E[i]['GWC']:.7}")
	for i in orbitals:
		print(f"state {i}; GW energy: {np.real(E[i]['GWC']):.7}\tstarting: {np.real(E[i]['HF-pert']):.7}\tcorrect: {np.real(E[i]['GW']):.7}")

	if False and args.self_energy:
		prefix, suffix = args.self_energy.split(",")
		filename_real, filename_imag = prefix + "-re_on_im" + suffix, prefix + "-im_on_im" + suffix
		orbital_i = int(suffix)
		qe_data = load_qe_se(filename_real, filename_imag, positive=True)
		z, s = qe_data["z"], qe_data["s"]
		fit_s = [get_model(orbital_i)(vz) for vz in z]
		z, s, fit_s = np.array(z), np.array(s), np.array(fit_s)
		fig, ax = plt.subplots(2)
		ax[0].plot(np.imag(z), np.real(s), label="reference s real")
		ax[0].plot(np.imag(z), np.imag(s), label="reference s imag")
		ax[0].plot(np.imag(z), np.real(fit_s), dashes=[2, 2], label="fit (qe) real")
		ax[0].plot(np.imag(z), np.imag(fit_s), dashes=[2, 2], label="fit (qe) imag")
		ax[0].plot(np.imag(z), np.real(qe_data["fit_imag"]), dashes=[3, 2], label="fit (qe param) re (im axis)")
		ax[0].plot(np.imag(z), np.imag(qe_data["fit_imag"]), dashes=[3, 2], label="fit (qe param) im (im axis)")	
		ax[0].legend()
		ax[1].plot(np.imag(z), np.real(qe_data["fit_real"]), label="fit (qe) re (re axis)")
		ax[1].plot(np.imag(z), np.imag(qe_data["fit_real"]), label="fit (qe) im (re axis)")
		y = list(map(get_model(orbital_i), z*(-1j)))
		ax[1].plot(np.imag(z), np.real(y), dashes=[2, 2], label="fit (qe param) re (re axis)")
		ax[1].plot(np.imag(z), np.imag(y), dashes=[2, 2], label="fit (qe param) im (re axis)")
		ax[1].legend()
		plt.show()

	# elif args.subparser == "gwwfit":
	# 	E = load_gww_energies(args.gww_out)

	# 	if args.qe:
	# 		prefix, suffix = args.self_energy.split(",")
	# 		filename_real = prefix + "-re_on_im" + suffix
	# 		filename_imag = prefix + "-im_on_im" + suffix
	# 		se_data = load_qe_se(filename_real, filename_imag)

	# 	def correlation(w):
	# 		for i in range(len(se_data["z"])-1):
	# 			if (w1 := np.imag(se_data["z"][i])) < np.real(w) and (w2 := np.imag(se_data["z"][i+1])) >= np.real(w):
	# 				return (se_data["fit_real"][i] * (w - w1) + se_data["fit_real"][i+1] * (w2 - w)) / (w2 - w1)
	
	# 	index = int(suffix)
	# 	offset = (E[5]["DFT"] + 0) / (2 * RY)
	# 	E0 = E[index]["DFT"]/RY - offset
	# 	E0 = complex(E0, 0)
	# 	E0 = E[index]["HF-pert"]/RY - offset + correlation(E0)

	# 	E = dyson_s(E[index]["HF-pert"] / RY - offset, E0, correlation)
	# 	E += offset
	# 	E *= RY
	# 	print(f"GW energy: {E}")