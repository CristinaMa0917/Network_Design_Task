import torch
import torch.nn as nn


class Test2Model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Test2Model, self).__init__()

        self.embedding = nn.Embedding(10, 10)
        self.conv1d = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=5, padding=2)
        self.conv1d2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)

        # self.linear = nn.Linear(10, out_channels)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0,2,1)

        x = self.conv1d(x)
        x = self.conv1d2(x)

        x = x.permute(0,2,1).squeeze(-1)

        return x


class Test1Model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Test1Model, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(-1).permute(0,2,1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1).squeeze(-1)

        return x