import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvCont(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv1d(2, 8, 7, stride = 3)
		self.conv2 = nn.Conv1d(8, 16, 5)
		self.conv3 = nn.Conv1d(16, 16, 5)
		self.pool = nn.MaxPool1d(2)
		self.fc1 = nn.Linear(16 * 6, 64)
		self.fc2 = nn.Linear(64, 481)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = torch.flatten(x, start_dim=1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

class ConvSECont(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv1d(in_channels = 2, out_channels = 32, kernel_size = 5, stride = 1)
		self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
		self.pool1 = nn.MaxPool1d(2)

		self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3)
		self.conv4 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3)
		self.pool2 = nn.MaxPool1d(2)

		self.fc1 = nn.Linear(13*64, 64)
		self.fc2 = nn.Linear(64, 481*2)
		self.dropout4 = nn.Dropout(p=0.4)
		self.dropout2 = nn.Dropout(p=0.25)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		x = F.relu(self.conv2(x))
		x = self.pool1(x)

		x = F.relu(self.conv3(x))
		x = self.pool1(x)
		x = F.relu(self.conv4(x))
		x = self.pool2(x)
		x = self.dropout4(x)
		x = torch.flatten(x, start_dim=1)
		x = F.relu(self.fc1(x))
		x = self.dropout2(x)
		x = self.fc2(x).reshape(-1, 2, 481)
		return x

class ConvSEContXL(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv1d(in_channels = 2, out_channels = 128, kernel_size = 5, stride = 1)
		self.pool1 = nn.MaxPool1d(2)
		self.conv2 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3)
		self.pool2 = nn.MaxPool1d(2)

		self.fc1 = nn.Linear(58*128, 512)
		self.fc2 = nn.Linear(512, 481*2)
		self.dropout4 = nn.Dropout(p=0.4)
		self.dropout2 = nn.Dropout(p=0.25)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		x = self.dropout4(x)
		x = torch.flatten(x, start_dim=1)
		x = F.relu(self.fc1(x))
		x = self.dropout2(x)
		x = self.fc2(x).reshape(-1, 2, 481)
		return x


class ConvSEContXXL(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv1d(in_channels = 2, out_channels = 128, kernel_size = 5, stride = 1)
		self.pool1 = nn.MaxPool1d(2)
		self.conv2 = nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 5)
		self.conv3 = nn.Conv1d(in_channels = 256, out_channels = 512, kernel_size = 3)
		self.fc1 = nn.Linear(54*256, 512)
		self.fc2 = nn.Linear(512, 481*2)
		self.dropout4 = nn.Dropout(p=0.4)
		self.dropout2 = nn.Dropout(p=0.25)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		x = F.relu(self.conv2(x))
		x = self.pool1(x)
		x = F.relu(self.conv3(x))
		x = self.pool1(x)
		x = self.dropout4(x)
		x = torch.flatten(x, start_dim=1)
		x = F.relu(self.fc1(x))
		x = self.dropout2(x)
		x = self.fc2(x).reshape(-1, 2, 481)
		return x

class ConvSEContX3L(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv1d(in_channels = 2, out_channels = 256, kernel_size = 9, stride = 1)
		self.pool1 = nn.MaxPool1d(2)
		self.conv2 = nn.Conv1d(in_channels = 256, out_channels = 512, kernel_size = 7)
		self.conv3 = nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 5)
		self.fc1 = nn.Linear(50*256, 1024)
		self.fc2 = nn.Linear(1024, 1024)
		self.fc3 = nn.Linear(1024, 481*2)
		self.dropout5 = nn.Dropout(p=0.5)
		self.dropout2 = nn.Dropout(p=0.25)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		x = F.relu(self.conv2(x))
		x = self.pool1(x)
		x = F.relu(self.conv3(x))
		x = self.pool1(x)
		x = torch.flatten(x, start_dim=1)
		x = self.dropout5(x)
		x = F.relu(self.fc1(x))
		x = self.dropout5(x)
		x = F.relu(self.fc2(x))
		x = self.dropout2(x)
		x = self.fc3(x).reshape(-1, 2, 481)
		return x