import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_model(x, y, model):
	# create two subplots
	fig, ax = plt.subplots(3, 1)
	# plot selfenergy.real() on the first subplot
	w = np.linspace(-20, 20, num=481)
	iw = np.linspace(0, 20, num=241)

	x_003 = x + torch.randn_like(x) * 0.003
	x_005 = x + torch.randn_like(x) * 0.005
	x_01 = x + torch.randn_like(x) * 0.01
	y_0 = model(x)
	y_003 = model(x_003)
	y_005 = model(x_005)
	y_01 = model(x_01)

	# plot real and imaginary parts of selfenergy.imag() on the second subplot
	ax[0].plot(iw, x.squeeze(0)[0], linewidth=0.5, color="blue", label="sigma = 0")
	ax[0].plot(iw, x_003.squeeze(0)[0], linewidth=0.5, color="magenta", label="sigma = 0.003")
	ax[0].plot(iw, x_005.squeeze(0)[0], linewidth=0.5, color="purple", label="sigma = 0.005")
	ax[0].plot(iw, x_01.squeeze(0)[0], linewidth=0.5, color="red", label="sigma = 0.01")
	ax[1].plot(iw, x.squeeze(0)[1], linewidth=0.5, color="blue")
	ax[1].plot(iw, x_003.squeeze(0)[1], linewidth=0.5, color="magenta")
	ax[1].plot(iw, x_005.squeeze(0)[1], linewidth=0.5, color="purple")
	ax[1].plot(iw, x_01.squeeze(0)[1], linewidth=0.5, color="red")
	ax[0].legend()
	# plot y on the second subplot
	ax[2].plot(w, y.squeeze(0), label="ideal spectrum", color="black", linewidth=0.5)
	ax[2].plot(w, y.squeeze(0), label="sigma = 0", color="cyan", linestyle="dashed", linewidth=0.5)
	ax[2].plot(w, y_003.squeeze(0), label="sigma = 0.003", color="magenta", linestyle="dashed", linewidth=0.5)
	ax[2].plot(w, y_005.squeeze(0), label="sigma = 0.005", color="purple", linestyle="dashed", linewidth=0.5)
	ax[2].plot(w, y_01.squeeze(0), label="sigma = 0.01", color="red", linestyle="dashed", linewidth=0.5)
	ax[2].legend()
	plt.show()