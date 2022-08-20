from turtle import color
import matplotlib.pyplot as plt
import numpy as np

def plot_model(x, y, output):
	# create two subplots
	fig, ax = plt.subplots(2, 1)
	# plot selfenergy.real() on the first subplot
	w = np.linspace(-20, 20, num=481)
	iw = np.linspace(0, 20, num=241)
	ax[1].plot(w, y, label="target rho", color="black", linestyle="-")
	# plot real and imaginary parts of selfenergy.imag() on the second subplot
	ax[0].plot(iw, x[0])
	ax[0].plot(iw, x[1])
	# plot y on the second subplot
	ax[1].plot(w, y, label="predicted rho", color="cyan", linestyle="dotted")
	plt.show()