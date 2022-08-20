import torch
import torch.nn as nn
from spectralgen import StorageSpectralDataset
import matplotlib.pyplot as plt
from models import ConvCont
from plotting import plot_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ConvCont().to(device)
criterion = nn.MSELoss()

testset = StorageSpectralDataset("train/1_test")
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
											shuffle=True, num_workers=0)

# load the model parameters from train/convcont.pt into model
model.load_state_dict(torch.load("train/convcont.pt"))
# evaluate the model on testset
with torch.no_grad():
	x, y = next(iter(testloader))
	outputs = model(x.to(device))
	plot_model(x.squeeze(0), y.squeeze(0), outputs)


with torch.no_grad():
	for i, data in enumerate(testset, 0):
		x, y = data
		outputs = model(x.to(device))
		loss = criterion(outputs, y.to(device))
		print(loss)
		break
