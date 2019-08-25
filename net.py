import torch
from torchvision import models
from torch import nn
from torch.autograd import Variable
from config import config
import numpy as np
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*61*61, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, config.classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, np.prod(x.size()[1:]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x
