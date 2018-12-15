import torch
import numpy as np
import pandas as pd
import torcheras
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import Task3Model
from dataset import TestDataset

model = Task3Model()

params = {
    'batch_size': 32,
    'lr': 0.01,
    'in_channels': 10,
    'out_channels': 10
}

dataset = TestDataset(params, 'data/task4_test_passage.csv', 'data/task4_test_query.csv', 'data/task4_test_label.csv')
dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False)

model = torcheras.Model(model, 'log/')
model.load_model('2018_7_2_21_55_42_888565', epoch=1)

model = model.model

result = []
flag = 0
for sample_batch in dataloader:
    x = sample_batch[0]
    embed = model(x)
    result.extend(F.log_softmax(embed).tolist())
    if flag % 100 == 0:
        print(flag)
    flag += 1

np.savetxt(open('data/embedding.csv', 'w'), np.array(result), delimiter=",")
