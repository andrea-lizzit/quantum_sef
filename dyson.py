#!/usr/bin/env python

import numpy as np
from qe_utils import load_gww_energies, load_gww_fit

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
		if abs(prev_E - E) < precision * abs(E0):
			pass
		if abs(prev_E - E) > abs(E0):
			raise RuntimeError("self-consistent calculation diverges")
	return E

if __name__ == "__main__":
	import csv, argparse, yaml
	import matplotlib.pyplot as plt
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest="subparser")
	p_loadparams = subparsers.add_parser("gwwparams")
	p_loadparams.add_argument("gww_out", help="filename of the gww output")
	
	p_a = subparsers.add_parser("a")
	p_a.add_argument("self_energy", help="prefix of name of the file containing the self-energy values")
	p_a.add_argument("file_index", help="index suffix of the file")
	p_a.add_argument("HF_energies", help="yaml file containing Hartree-Fock energies of HOMO, LUMO, and target")
	args = parser.parse_args()
	
	if args.subparser == "gwwparams":
		print("starting dyson from gww")
		params = load_gww_fit(args.gww_out)[4]
		E = load_gww_energies(args.gww_out)
		def correlation(z):
			return multipole(z, params)
		offset = (E[4]["HF-pert"] + E[5]["HF-pert"]) / 2
		E0 = E[5]["HF-pert"] - offset
		E0 = complex(E0, 0)

		print(f"starting dyson self-consistent calculation with E0={E0}")
		E = dyson(E0, correlation)
		E += offset
		print(f"GW energy: {E}")
	elif args.subparser == "a":
		with open(args.HF_energies, "r") as fd:
			energies = yaml.safe_load(fd)
			e_HOMO, e_LUMO, e_target = energies["HOMO"], energies["LUMO"], energies["target"]
	
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
		print(f"GW energy: {E}")