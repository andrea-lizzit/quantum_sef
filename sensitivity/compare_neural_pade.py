import numpy as np
import torch
from pade.model_pade import PadeModel
from neural.spectralgen import StorageSpectralDataset
from neural.storagemanager import StorageManager
import matplotlib.pyplot as plt
from jax.config import config
config.update("jax_enable_x64", True)

storage = StorageManager()
z = np.linspace(0, 20, 241)
testset = StorageSpectralDataset(storage.datasets()[-1].test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
											shuffle=True, num_workers=0)

# create a PadeModel on this data
z, s = se["z"], se["s"]
M, N = range(8, 80, 8), [8, 16, 74, 76, 78, 80]
z, s, M, N = np.array(z), np.array(s), np.array(M), np.array(N)
# models = AverageModel.get_models(z, s, M, N, precise_iter=0)
# w = np.imag(z)

# create an AverageSimilarModel
# model = AverageSimilarModel(models, w)

args = 80, 16
model = PadeModel.from_specs(z, s, *args)
def fit_model(model, z, s):
	return PadeModel.from_specs(z, s, *args)

res = sensitivity(PadeModel, model.pparams, z, fit_model, Ei, qedir.offset)

for e in res.res:
	print(f"{np.real(e):.7}")
print(res.summary)
plt.show()