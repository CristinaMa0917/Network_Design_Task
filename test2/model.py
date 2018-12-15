import torch
import torch.nn as nn
import numpy as np

'''
class Task3Model(nn.Module):
    def __init__(self):
        super(Task3Model, self).__init__()

        self.embedding = nn.Embedding(11, 8)

        self.conv1d = nn.Conv1d(in_channels=8, out_channels=10, kernel_size=3, padding=1)
        self.conv1d2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=3, padding=1)

        self.conv1d_query = nn.Conv1d(in_channels=8, out_channels=10, kernel_size=1)

        self.linear1 = nn.Linear(11,10)
        self.linear2 = nn.Linear(10 * 10, 10)

    def forward(self, x):
        x, x_query = x

        x = self.embedding(x)
        x_query = self.embedding(x_query)

        x_query = x_query.repeat(1,10,1)
        x_query = x_query.permute(0,2,1)
        x_query = self.conv1d_query(x_query)

        x = torch.cat([x.permute(0,2,1), x_query], 1)

        x = x.permute(0,2,1)
        x = self.conv1d(x)
        x = self.conv1d2(x)

        x = x.permute(0, 2, 1)

        x = self.linear1(torch.cat([x, x_query], 2))
        return self.linear2(x.view(32, -1)).squeeze(-1)
'''

class Task3Model(nn.Module):
    def __init__(self):
        super(Task3Model, self).__init__()
        
        hidden_dim = 16
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(11, 8)
        # self.projection = nn.Linear(8, 1)
        
        # self.attn = nn.Linear(10, 10)

        self.rnn = nn.LSTM(16, hidden_dim, 2, bidirectional=True)
        self.h = (torch.randn(4, 10, hidden_dim), torch.randn(4, 10, hidden_dim))
        
        # self.linear = nn.Linear(64 + 8, 1)
        # self.out = nn.Linear(72, 1)
        self.out = nn.Linear(10 * hidden_dim * 2, 10)
        # self.out2 = nn.Linear(10, 10)
          
    def forward(self, x):
        x, x_query = x

        x = self.embedding(x)
        x_query = self.embedding(x_query)

        # x = self.projection(x)
        # x_query = self.projection(x_query)
    
        x_query = x_query.repeat(1, 10, 1)
        x = torch.cat([x, x_query], 2)

        self.h = (torch.randn(4, 10, self.hidden_dim), torch.randn(4, 10, self.hidden_dim))
        output, hn = self.rnn(x, self.h)

        output = output.view(output.shape[0], -1)
        return self.out(output)
        # return self.out2(self.out(output))
