import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from tqdm import tqdm, trange
from spectralgen import SpectralDataset, StorageSpectralDataset
from models import ConvCont
batch_size = 256

model = ConvCont()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
	model.cuda()
else:
	print("CUDA not available. Running on CPU")

summary(model, input_size=(batch_size, 2, 241))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

try:
	testset = StorageSpectralDataset("train/1_test")
except FileNotFoundError:
	testset = SpectralDataset(100)
	testset.save("train/1_test/")

try:
	trainset = StorageSpectralDataset("train/1_train")
except FileNotFoundError:
	trainset = SpectralDataset(1000)
	trainset.save("train/1_train/")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
											shuffle=True, num_workers=0)


print_every = 2000 // batch_size
epochs = 30

bar = trange(30)
for epoch in bar:
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        x, y = data
        x += torch.randn_like(x) * 0.01

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(x.to(device))
        loss = criterion(outputs, y.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print(running_loss / (i%print_every+1))
        if i % print_every == print_every-1:    # print every 2000 mini-batches
            bar.set_postfix(loss='{:.3f}'.format(running_loss / print_every))
            running_loss = 0.0

print('Finished Training')

PATH = 'train/convcont.pt'
torch.save(model.state_dict(), PATH)