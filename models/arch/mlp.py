import torch
import numpy as np
from torch import nn, tensor
from typing import *


class MLP(nn.Module):
    
    def __init__(self, input_shape: List[int], num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=np.prod(input_shape[1:]), out_features=512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)
        
        
    def forward(self, x: tensor) ->tensor:
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x