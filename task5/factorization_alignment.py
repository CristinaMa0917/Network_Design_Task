import torch
import torch.nn as nn
from factorization import Factorization

class FactorizationAlignment(nn.Module):
    def __init__(self, in_features, k):
        super(FactorizationAlignment, self).__init__()
        self.factorization1 = Factorization(in_features*2, k)
        self.factorization2 = Factorization(in_features, k)
        self.factorization3 = Factorization(in_features, k)
    
    def forward(self, a, b):
        # concat
        concat_ab = torch.cat([a, b], dim=-1)
        sub_ab = a - b
        dot_ab = torch.einsum('bij,bij->bij', (a, b))

        c = self.factorization1(concat_ab)
        s = self.factorization2(sub_ab)
        m = self.factorization3(dot_ab)

        return c, s, m
