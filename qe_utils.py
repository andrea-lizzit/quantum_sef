import numpy as np
import csv, re
from common import *


def load_gww_fit(filename):
	with open(filename, "r") as fd:
		gww_out = fd.readlines()
	gww_out = "".join(gww_out)
	matches = re.findall("FIT state :.+\n (?:FIT.+\n )+", gww_out)
	params = []
	for match in matches:
		orbital_i, j = re.match("FIT state :\s+(\d)\s+(\d)", match).group(1, 2)
		orbital_i = float(orbital_i)
		a_0 = re.search("FIT\s+a_0: \((.*?),(.*?)\)", match).group(1, 2)
		a_0 = complex(float(a_0[0]), float(a_0[1]))
		poles_list = re.findall("FIT\s+a:\s+\d\s+\((.*?),(.*?)\)\s+FIT\s+b:\s+\d\s+\((.*?),(.*?)\)\n", match)
		poles = []
		for pole_list in poles_list:
			a = complex(float(pole_list[0]), float(pole_list[1]))
			b = complex(float(pole_list[2]), float(pole_list[3]))
			poles.append(Pole(a, b))
		params.append(MPParams(a_0, poles))
	return params

def load_gww_energies(filename):
	pattern_energy = "State:\s+(\d)DFT\s+:\s+(\S+)+ GW-PERT\s+:\s+(\S+) GW\s+:\s+(\S+) HF-pert\s+:\s+(\S+)\n"
	with open(filename, "r") as fd:
		gww_out = fd.readlines()
	gww_out = "".join(gww_out)
	states = re.findall(pattern_energy, gww_out)
	E = dict()
	for state in states:
		orbital_i = int(state[0])
		E[orbital_i] = {"DFT": float(state[1]), "GW-PERT": float(state[2]), "GW": float(state[3]), "HF-pert": float(state[4])}
	return E		

def load_qe_se(file_real, file_im, n_fit=None):
	"Load the self energy values from a quantum-espresso file"
	real, imag = [], []
	with open(file_real, "r") as fd:
		realreader = csv.reader(fd, delimiter=' ', skipinitialspace=True)
		for i, row in enumerate(realreader):
			if float(row[0]) < 0:
				continue
			row = list(map(float, row))
			real.append(row)
			if i+1 == n_fit:
				pass

	rfreq, rqefitim, rse, rqefitr = list(zip(*real))

	with open(file_im, "r") as fd:
		imreader = csv.reader(fd, delimiter=' ', skipinitialspace=True)
		for i, row in enumerate(imreader):
			if float(row[0]) < 0:
				continue
			row = list(map(float, row))
			imag.append(row)
			if i+1 == n_fit:
				pass
	
	ifreq, iqefitim, ise, iqefitr = list(zip(*real))
	if rfreq != ifreq:
		raise RuntimeError("error: real freq and im freq are different")
	freq = rfreq
	qefitim = list(map(complex, rqefitim, iqefitim))
	se = list(map(complex, rse, ise))
	qefitr = list(map(complex, rqefitr, iqefitr))
	z = list(map(lambda i: complex(0, i), freq))
	return {"z": z, "s": se, "fit_imag": qefitim, "fit_real": qefitr}