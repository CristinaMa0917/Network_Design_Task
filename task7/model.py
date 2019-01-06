import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from constants import device

class TaskModel(nn.Module):
    def __init__(self, params):
        super(TaskModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(3, stride=2)
        self.prediction = nn.Linear(1152, 1)
        # self.prediction1 = nn.Linear(8,1)
    
    def forward(self, x):
        x = x.unsqueeze(-1).permute(0,3,1,2)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = x.view(x.shape[0], -1)
        
        pred = self.prediction(x)
        
        return pred.squeeze(-1)
