


def multipole(z, params):
	"""Multipole function to fit."""
	v = params.bias
	for pole in params.poles:
		v += pole.a / (z - pole.b)
	return v

def dyson(E0, correlation, precision=0.01):
	E = E0
	while True:
		print(f"new E: {E}")
		E = E0 + correlation(E0)
		if abs(E0 - E) < precision * abs(E0):
			break
		if abs(E0 - E) > abs(E0):
			raise RuntimeError("self-consistent calculation diverges")
		E0 = E
	return E

if __name__ == "__main__":
	import csv, argparse, yaml
	import matplotlib.pyplot as plt
	parser = argparse.ArgumentParser()
	parser.add_argument("self_energy", help="name of the file containing the self-energy values")
	parser.add_argument("HF_energies", help="yaml file containing Hartree-Fock energies of HOMO, LUMO, and target")
	args = parser.parse_args()
	
	with open(args.HF_energies, "r") as fd:
		energies = yaml.safe_load(fd)
		e_HOMO, e_LUMO, e_target = energies["HOMO"], energies["LUMO"], energies["target"]
	
	real, rfreq = [], []
	with open(args.self_energy, "r") as fd:
		realreader = csv.reader(fd, delimiter=' ', skipinitialspace=True)
		for i, row in enumerate(realreader):
			rfreq.append(float(row[0]))
			real.append(float(row[2]))
	
	def correlation(w):
		for i in range(len(rfreq)-1):
			if w1 := rfreq[i] < w and (w2 := rfreq[i+1]) >= w:
				return (real[i] * (w - w1) + real[i+1] * (w2 - w)) / (w2 - w1)

				

	
	offset = (e_LUMO + e_HOMO) / 2
	E0 = e_target - offset


	E = dyson(E0, correlation)
	E += offset
	print(f"GW energy: {E}")