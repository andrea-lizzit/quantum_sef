#!/usr/bin/env python

import numpy as np
from qe_utils import load_gww_energies, load_gww_fit, load_qe_se

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
		print(f"self energy {correlation(E)}")
		if abs(prev_E - E) < precision * abs(E0):
			pass
		if abs(prev_E - E) > 10*abs(E0):
			raise RuntimeError("self-consistent calculation diverges")
	return E

if __name__ == "__main__":
	import csv, argparse, yaml
	import matplotlib.pyplot as plt
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest="subparser")
	p_loadparams = subparsers.add_parser("gwwparams")
	p_loadparams.add_argument("gww_out", help="filename of the gww output")
	p_loadparams.add_argument("--self_energy", "-s", help="qe self-energy file for comparison")
	p_a = subparsers.add_parser("a")
	p_a.add_argument("self_energy", help="prefix of name of the file containing the self-energy values")
	p_a.add_argument("file_index", help="index suffix of the file")
	p_a.add_argument("HF_energies", help="yaml file containing Hartree-Fock energies of HOMO, LUMO, and target")
	args = parser.parse_args()
	
	if args.subparser == "gwwparams":
		print("starting dyson from gww")
		E = load_gww_energies(args.gww_out)

		def correlation(orbital_i):
			params = load_gww_fit(args.gww_out)[orbital_i-1]
			def inner(z):
				if np.real(z) > 0:
					return multipole(z, params)
				else:
					sigma = multipole(np.conj(z), params) # like in qe
					return np.conj(sigma)
			return inner

		offset = (E[5]["DFT"] + 0) / (2 * RY)
		print(f'offset {offset} = {E[5]["DFT"]} + {0} / 2RY')

		def GW(qe_E, correlation):
			E0 = qe_E["DFT"]/RY - offset
			E0 = complex(E0, 0)
			E0 = qe_E["HF-pert"]/RY - offset + correlation(E0)

			E = dyson_s(qe_E["HF-pert"] / RY - offset, E0, correlation)
			E += offset
			E *= RY
			return E

		for i in range(1, len(E)+1):
			E[i]["GWC"] = GW(E[i], correlation(i))
			print(f"state {i}; GW energy: {E[i]['GWC']:.7}")
		for i in range(1, len(E)+1):
			print(f"state {i}; GW energy: {np.real(E[i]['GWC']):.7}")

		if args.self_energy:
			prefix, suffix = args.self_energy.split(",")
			filename_real, filename_imag = prefix + "-re_on_im" + suffix, prefix + "-im_on_im" + suffix
			orbital_i = int(suffix)
			qe_data = load_qe_se(filename_real, filename_imag, positive=True)
			z, s = qe_data["z"], qe_data["s"]
			fit_s = [correlation(orbital_i)(vz) for vz in z]
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
			y = list(map(correlation(orbital_i), z*(-1j)))
			ax[1].plot(np.imag(z), np.real(y), dashes=[2, 2], label="fit (qe param) re (re axis)")
			ax[1].plot(np.imag(z), np.imag(y), dashes=[2, 2], label="fit (qe param) im (re axis)")
			ax[1].legend()
			plt.show()

	elif args.subparser == "a":
		with open(args.HF_energies, "r") as fd:
			energies = yaml.safe_load(fd)
			e_HOMO, e_LUMO, e_target = energies["HOMO"]*RY, energies["LUMO"]*RY, energies["target"]*RY
	
		real, rfreq = [], []
		filename_real = args.self_energy + "-re_on_im" + args.file_index
		filename_imag = args.self_energy + "-im_on_im" + args.file_index
		with open(filename_real, "r") as fd:
			realreader = csv.reader(fd, delimiter=' ', skipinitialspace=True)
			for i, row in enumerate(realreader):
				rfreq.append(float(row[0]))
				real.append(float(row[3]))

		def correlation(w):
			for i in range(len(rfreq)-1):
				if w1 := rfreq[i] < w and (w2 := rfreq[i+1]) >= w:
					return (real[i] * (w - w1) + real[i+1] * (w2 - w)) / (w2 - w1)
	
		offset = (e_LUMO + e_HOMO) / 2
		E0 = e_target - offset
		E0 = np.complex(E0, 0)

		E = dyson(E0, correlation)
		E += offset
		E *= RY
		print(f"GW energy: {E}")