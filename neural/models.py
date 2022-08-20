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