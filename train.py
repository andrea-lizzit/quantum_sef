import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
import argparse
from tqdm import trange
from neural.xydataset import XYDataset
from neural.xydataset import generate as xygen
from neural.models import ConvCont, ConvSECont, ConvSEContX3L
from neural.plotting import plot_model
from neural.storagemanager import StorageManager

storage = StorageManager()

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", action="store_true", help="load the last saved model")
parser.add_argument("--version", type=int, default=4, help="version of the model")
args = parser.parse_args()

batch_size = 32

# load model and setup SummaryWriter
date_signature = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model = ConvSEContX3L()


if args.load_model:
    model_path = storage.last_model()
    print("Loading model from {}".format(model_path))
    model.load_state_dict(torch.load(str(model_path)))


writer = SummaryWriter(storage.tensorboard_logdir() / storage.new_session().name)

# choose optimal device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
	model.cuda()
else:
	print("CUDA not available. Running on CPU")

# loss
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# load data
try:
	testset = XYDataset.load(storage.datasets()[-1].test)
except (IndexError, FileNotFoundError):
	testset = xygen.se_dataset(5000)
	testset.save(storage.new_dataset() / "test")

try:
	trainset = XYDataset.load(storage.datasets()[-1].train)
except (IndexError, FileNotFoundError):
	trainset = xygen.se_dataset(70000)
	trainset.save(storage.new_dataset() / "train")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											shuffle=True, num_workers=0, drop_last=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
											shuffle=True, num_workers=0)
testiter = iter(testloader)
print(f"length of testiter: {len(testiter)}")

# output model info
summary(model, input_size=(batch_size, 2, 241))
writer.add_graph(model, iter(trainloader).next()[0].to(device))

# hyperparameters
print_every = 100
epochs = 3000

# training loop
bar = trange(epochs)
val_loss = 0
for epoch in bar:
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        x, y = data
        # x += torch.randn_like(x) * 0.04 * 0.01

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

    # save model every 5 epochs
    if epoch % 5 == 0:
        if not storage.new_session().exists():
            storage.new_session().mkdir()
        torch.save(model.state_dict(), str(storage.new_session() / "model_{}.pt".format(epoch)))


# save final model
print('Finished Training')
torch.save(model.state_dict(), str(storage.new_session() / "model.pt"))
print("Saved model to {}".format(str(storage.new_session() / "model.pt")))