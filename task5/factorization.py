import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import device

class Factorization(nn.Module):
    def __init__(self, in_features, k): 
        super(Factorization, self).__init__()     

        self.in_features = in_features
        self.w = nn.Linear(in_features=in_features, out_features=1)
        self.v = nn.Parameter(torch.Tensor(in_features, k).uniform_())

        nn.init.xavier_normal_(self.w.weight)
        nn.init.xavier_normal_(self.v)

    def forward(self, x):
        # L(.)
        l = self.w(x)

        # P(.)
        batch_size = x.shape[0]
        v = torch.mm(self.v, self.v.t())
        v = v.unsqueeze(-1).permute(2,0,1).repeat(batch_size, 1, 1)
        result = torch.bmm(torch.bmm(x, v), x.permute(0, 2, 1))

        p = torch.zeros(batch_size, x.shape[1]).to(device)
        for i in range(batch_size):
            p[i] = ((result[i] - torch.diag(torch.diag(result[i]))) / 2).sum(dim=-1)

        return l + p.unsqueeze(-1)