import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pandas as pd
import numpy as np

from model import Task5Model
from dataset import TaskDataset

import torcheras

def main(params):

    model = Task5Model(params)
    dataset = TaskDataset(params, 'data/task5_2_train_paragraph1.csv', 'data/task5_2_train_paragraph2.csv', 'data/task5_2_train_label.csv')

    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    model = torcheras.Model(model, 'log/')
    model.compile(loss_fn, optimizer, metrics=['binary_acc'])

    writer = SummaryWriter()

    result = []
    def callback(epoch, i_batch, metrics_result):
        result.append(metrics_result['binary_acc'])

    model.fit(dataloader, epochs=1, batch_callback=callback)
    print(sum(result[-100:]))

    p1 = np.array(pd.read_csv('data/task5_2_test_paragraph1.csv', header=None), dtype=np.int)
    p2 = np.array(pd.read_csv('data/task5_2_test_paragraph2.csv', header=None), dtype=np.int)

    results = []
    for i in range(p1.shape[0]):
        result = model.model.forward((torch.LongTensor([p1[i]]), torch.LongTensor([p2[i]])))
        if result > 0.5:
            results.append(1)
        else: 
            results.append(0)
    print(results)

if __name__ == '__main__':
    params = {
        'in': 30,
        'embed_dim': 8,
        'batch_size': 32,
        'lr': 0.0001,
        'out_dim':2
    }
    main(params)