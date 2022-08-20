import torch
import torch.nn as nn
from spectralgen import StorageSpectralDataset
import matplotlib.pyplot as plt
from pathlib import Path
from models import ConvCont
from plotting import plot_model

device = torch.device("cpu") # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ConvCont().to(device)
criterion = nn.MSELoss()

testset = StorageSpectralDataset("train/2_test")
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
											shuffle=True, num_workers=0)

# load the model parameters from train/convcont$num.pt into model
# use the last model in the list
model_path = list(Path("train").glob("convcont*.pt"))[-1]
print("Loading model from {}".format(model_path))
model.load_state_dict(torch.load(str(model_path)))
# evaluate the model on testset
with torch.no_grad():
	x, y = next(iter(testloader))
	x = x.to(device)
	y = y.to(device)
	plot_model(x, y, model)


# with torch.no_grad():
# 	for i, data in enumerate(testset, 0):
# 		x, y = data
# 		outputs = model(x.to(device))
# 		loss = criterion(outputs, y.to(device))
# 		print(loss)
# 		break
