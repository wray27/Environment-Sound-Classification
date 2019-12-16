from dataset import UrbanSound8KDataset
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
import torch
from torchvision.transforms import Compose
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count
        self.dropout = nn.Dropout(p=dropout)

        self.normaliseConv1 = nn.BatchNorm2d(
            num_features=32
        )
        self.normaliseConv2 = nn.BatchNorm2d(
            num_features=64
        )
        self.normaliseFC = nn.BatchNorm1d(
            num_features=1024
        )


        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        ## TASK 2-1: Define the second convolutional layer and initialise its parameters
        self.conv2 = nn.Conv2d(
             in_channels=32,
             out_channels=64,
             kernel_size=(5, 5),
             padding=(2, 2),
        )
        self.initialise_layer(self.conv2)
        ## TASK 3-1: Define the second pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        ## TASK 5-1: Define the first FC layer and initialise its parameters
        self.fc1 = nn.Linear(4096, 1024)
        self.initialise_layer(self.fc1)
        ## TASK 6-1: Define the last FC layer and initialise its parameters
        self.fc2 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.normaliseConv1(self.conv1(images)))
        x = self.pool1(x)
        x = F.relu(self.normaliseConv2(self.conv2(x)))
        x = self.pool2(x)
        ## TASK 4: Flatten the output of the pooling layer so it is of shape
        ##         (batch_size, 4096)
        x = torch.flatten(x, 1)
        ## TASK 5-2: Pass x through the first fully connected layer
        x = F.relu(self.normaliseFC(self.fc1(self.dropout(x))))
        ## TASK 6-2: Pass x through the last fully connected layer
        x = self.fc2(self.dropout(x))
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


mode = "MLMC"
train_loader = torch.utils.data.DataLoader(
      UrbanSound8KDataset('UrbanSound8K_train.pkl', mode),
      batch_size=32, shuffle=True,
      num_workers=8, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
     UrbanSound8KDataset('UrbanSound8K_test.pkl', mode),
     batch_size=32, shuffle=False,
     num_workers=8, pin_memory=True)


# print(UrbanSound8KDataset('UrbanSound8K_test.pkl', mode).__getitem__(0))


for i, (input, target, filename) in enumerate(train_loader):
    pass
#           training code



for i, (input, target, filename) in enumerate(val_loader):
    pass
#           validation code
