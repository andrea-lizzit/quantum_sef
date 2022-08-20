from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from spectralgen import SpectralDataset, StorageSpectralDataset
from models import ConvCont
from plotting import plot_model

batch_size = 512
load_model = True

# load model and setup SummaryWriter
date_signature = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model = ConvCont()


if load_model:
    model_dir = list(Path("train").glob("v1_*"))[-1]
    model_path = list(model_dir.glob("*.pt"))[-1]
    print("Loading model from {}".format(model_path))
    model.load_state_dict(torch.load(str(model_path)))

current_path = Path("train") / "v1_{}".format(date_signature)

writer = SummaryWriter('runs')

# choose optimal device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
	model.cuda()
else:
	print("CUDA not available. Running on CPU")

# loss
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# load data
try:
	testset = StorageSpectralDataset("train/2_test")
except FileNotFoundError:
	testset = SpectralDataset(100)
	testset.save("train/2_test/")

try:
	trainset = StorageSpectralDataset("train/2_train")
except FileNotFoundError:
	trainset = SpectralDataset(1000)
	trainset.save("train/2_train/")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
											shuffle=True, num_workers=0)
testiter = iter(testloader)
print(f"lenght of testiter: {len(testiter)}")

# output model info
summary(model, input_size=(batch_size, 2, 241))
writer.add_graph(model, iter(trainloader).next()[0].to(device))

# hyperparameters
print_every = 100
epochs = 30

# training loop
bar = trange(10000)
val_loss = 0
for epoch in bar:
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        x, y = data
        x += torch.randn_like(x) * 0.01

        optimizer.zero_grad()

        outputs = model(x.to(device))
        loss = criterion(outputs, y.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % print_every == print_every-1:
            bar.set_postfix(loss='{:.5f}'.format(running_loss / print_every),
                            val_loss='{:.5f}'.format(val_loss))
            writer.add_scalar('training_loss',
                            running_loss / print_every,
                            epoch * len(trainloader) + i)
            running_loss = 0.0
        
    with torch.no_grad():
        try:
            val_x, val_y = testiter.next()
        except StopIteration:
            testiter = iter(testloader)
            val_x, val_y = testiter.next()
        val_x, val_y = val_x.to(device), val_y.to(device)
        val_loss = criterion(model(val_x), val_y).item()
        writer.add_scalar('validation_loss', val_loss, epoch * len(trainloader))

    if epoch % 10 == 0:
        with torch.no_grad():
            writer.add_figure("predictions", plot_model(val_x[0].unsqueeze(0), val_y[0].unsqueeze(0), model, plot=False), global_step = epoch * len(trainloader))

    # save model every 50 epochs
    if epoch % 50 == 0:
        if not current_path.exists():
            current_path.mkdir()
        torch.save(model.state_dict(), str(current_path / "model_{}.pt".format(epoch)))


# save final model
print('Finished Training')
torch.save(model.state_dict(), str(current_path / "model.pt"))
print("Saved model to {}".format(str(current_path / "model.pt")))