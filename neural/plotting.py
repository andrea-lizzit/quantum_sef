import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_model(x, y, model, plot=True):
	# create two subplots
	fig, ax = plt.subplots(2, 2, figsize=(9, 9))
	# plot selfenergy.real() on the first subplot
	w = np.linspace(-20, 20, num=481)
	iw = np.linspace(0, 20, num=241)
	x_003 = x + torch.randn_like(x) * 0.04 * 0.003
	x_005 = x + torch.randn_like(x) * 0.04 * 0.005
	x_01 = x + torch.randn_like(x)  * 0.04 * 0.01
	y_0 = model(x)
	y_003 = model(x_003)
	y_005 = model(x_005)
	y_01 = model(x_01)

	# plot real and imaginary parts of selfenergy.imag() on the second subplot
	ax[0, 0].set_ylabel("Re[Σ(ix)] (Ry)")
	ax[0, 0].set_xlabel("x (Ry)")
	ax[0, 0].plot(iw, x.squeeze(0)[0].cpu(), linewidth=0.5, color="blue", label="sigma = 0")
	ax[0, 0].plot(iw, x_003.squeeze(0)[0].cpu(), linewidth=0.5, color="magenta", label="sigma = 0.003")
	ax[0, 0].plot(iw, x_005.squeeze(0)[0].cpu(), linewidth=0.5, color="purple", label="sigma = 0.005")
	ax[0, 0].plot(iw, x_01.squeeze(0)[0].cpu(), linewidth=0.5, color="red", label="sigma = 0.01")
	ax[0, 1].set_ylabel("Im[Σ(ix)] (Ry)")
	ax[0, 0].set_xlabel("x (Ry)")
	ax[0, 1].plot(iw, x.squeeze(0)[1].cpu(), linewidth=0.5, color="blue")
	ax[0, 1].plot(iw, x_003.squeeze(0)[1].cpu(), linewidth=0.5, color="magenta")
	ax[0, 1].plot(iw, x_005.squeeze(0)[1].cpu(), linewidth=0.5, color="purple")
	ax[0, 1].plot(iw, x_01.squeeze(0)[1].cpu(), linewidth=0.5, color="red")
	ax[0, 0].legend()
	# plot y on the third subplot
	if len(y.shape) == 2:
		ax[1, 1].plot(w, y.squeeze(0).cpu(), label="ideal spectrum", color="black", linewidth=0.5)
		ax[1, 1].plot(w, y_0.squeeze(0).cpu(), label="sigma = 0", color="cyan", linestyle="dashed", linewidth=0.5)
		ax[1, 1].plot(w, y_003.squeeze(0).cpu(), label="sigma = 0.003", color="magenta", linestyle="dashed", linewidth=0.5)
		ax[1, 1].plot(w, y_005.squeeze(0).cpu(), label="sigma = 0.005", color="purple", linestyle="dashed", linewidth=0.5)
		ax[1, 1].plot(w, y_01.squeeze(0).cpu(), label="sigma = 0.01", color="red", linestyle="dashed", linewidth=0.5)
	else:
		ax[1, 0].set_ylabel("Re[Σ(x)] (Ry)")
		ax[1, 0].set_xlabel("x (Ry)")
		ax[1, 0].plot(w, y[0, 0].cpu(), label="exact", color="black", linewidth=0.5)
		ax[1, 0].plot(w, y_0[0, 0].cpu(), label="sigma = 0", color="#003cff", linestyle="dashed", linewidth=0.5)
		ax[1, 0].plot(w, y_003[0, 0].cpu(), label="sigma = 0.003", color="magenta", linestyle="dashed", linewidth=0.5, alpha = 0.3)
		ax[1, 0].plot(w, y_005[0, 0].cpu(), label="sigma = 0.005", color="purple", linestyle="dashed", linewidth=0.5, alpha = 0.3)
		ax[1, 0].plot(w, y_01[0, 0].cpu(), label="sigma = 0.01", color="red", linestyle="dashed", linewidth=0.5, alpha = 0.3)
		ax[1, 1].set_ylabel("Im[Σ(x)] (Ry)")
		ax[1, 1].set_xlabel("x (Ry)")
		ax[1, 1].plot(w, y[0, 1].cpu(), color="black", linewidth=0.5)
		ax[1, 1].plot(w, y_0[0, 1].cpu(), color="#003cff", linestyle="dashed", linewidth=0.5)
		ax[1, 1].plot(w, y_003[0, 1].cpu(), color="magenta", linestyle="dashed", linewidth=0.5, alpha = 0.3)
		ax[1, 1].plot(w, y_005[0, 1].cpu(), color="purple", linestyle="dashed", linewidth=0.5, alpha = 0.3)
		ax[1, 1].plot(w, y_01[0, 1].cpu(), color="red", linestyle="dashed", linewidth=0.5, alpha = 0.3)
	ax[1, 0].legend()
	if plot:
		plt.show()
	else:
		return fig