import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from constants import device

from factorization_alignment import FactorizationAlignment
from self_attention import SelfAttention

class Task5Model(nn.Module):
    def __init__(self, params):
        super(Task5Model, self).__init__()

        # params
        self.embed_dim = params['embed_dim']
        self.hidden_dim = 16
        self.k = 12
        self.dropout = 0.1

        # network
        self.embedding = nn.Embedding(params['in'], self.embed_dim)
        self.rnn = nn.LSTM(16, self.hidden_dim, 2)
        self.hidden = (torch.randn(2, 10, self.hidden_dim), torch.randn(2, 10, self.hidden_dim))

        self.pred = nn.Linear(80*2, 1)
        # self.pred = nn.Linear(100, 1)
        
    def forward(self, x):
        p1, p2 = x
        sent1 = self.embedding(p1)
        sent2 = self.embedding(p2)

        # inter attn
        inter_alignment = torch.bmm(sent1, sent2.permute(0, 2, 1))
        beta = torch.bmm(F.softmax(inter_alignment, dim=2), sent2)
        alpha = torch.bmm(F.softmax(inter_alignment, dim=1).permute(0, 2, 1), sent1)

        rnn_input = torch.cat([beta, alpha], dim=-1)
        pred_input, hn = self.rnn(rnn_input, self.hidden)

        # pred_input = inter_alignment
        return self.pred(pred_input.view(pred_input.shape[0], -1))