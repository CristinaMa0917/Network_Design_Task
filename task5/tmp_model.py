import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from constants import device

class Task5Model(nn.Module):
    def __init__(self, params):
        super(Task5Model, self).__init__()
        
        hidden_dim = 2
        
        # self.embedding = 
        self.rnn = nn.LSTM(1, hidden_dim, 2)
        self.h = (torch.randn(2, 4, hidden_dim), torch.randn(2, 4, hidden_dim))

        
    def forward(self, x):
        
        output, hn = self.rnn(x, self.h)

        return output
