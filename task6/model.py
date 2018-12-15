import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from constants import device

# class RNN()

class Task5Model(nn.Module):
    def __init__(self, params):
        super(Task5Model, self).__init__()
        
        self.hidden_dim = 1
        self.embedding_dim = 8
        
        self.embedding = nn.Embedding(2, self.embedding_dim)
        # self.embedding.weight.requires_grad = False
        self.rnn = nn.LSTM(8, self.hidden_dim, 1)
        # self.h = (torch.randn(1, 20, self.hidden_dim), torch.randn(1, 20, self.hidden_dim))
        # self.h = torch.ones(32, self.hidden_dim) * -1
        self.h = torch.randn(32, self.hidden_dim)

        # self.gated_x = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.linear_x = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.linear_h = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.output = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, x):
        original_x = x

        x = self.embedding(x)
        h = self.h
        for i in range(x.shape[1]):
            # print(original_x.shape)
            linear_x = self.linear_x(x[:,i,:])
            # print(original_x[:1,i])
            # print(linear_x[:1,0])
            h = F.tanh(linear_x) / F.tanh(h)
            # print(h)
            # input()
        # print('1', F.sigmoid(h))
        return h

    # def forward(self, x):
    #     x = self.embedding(x)

    #     for i in range(x.shape[1]):
    #         self.h = torch.sin(self.linear_h(self.h) + self.linear_x(x[:,i,:]))

    #     return self.h