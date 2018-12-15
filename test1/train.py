
import torch
from model import Test1Model, Test2Model
from dataset import TestDataset

from torch.utils.data import DataLoader

import torch.nn.functional as F


def metrics_fn(y_pred, y_true):
    y_true = y_true.contiguous()
    
    y_pred = F.softmax(y_pred)
    y_pred = torch.max(y_pred, 1)[1]
    return torch.mean(y_true.eq(y_pred.long()).float(), 0)


def main(params):

    model = Test2Model(params['in_channels'], params['out_channels'])
    dataset = TestDataset(params, 'data/task2_data.csv', 'data/task2_label.csv')

    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    result = []

    for i_batch, sample_batch in enumerate(dataloader):

        model.zero_grad()

        x = sample_batch[0]
        y = sample_batch[1].long()
        y = y.squeeze(1)
        y_pred = model.forward(x)

        loss = loss_fn(y_pred, y)
        loss.backward()

        optimizer.step()

        metric = metrics_fn(y_pred, y)

        result.append(metric)

        print(i_batch, metric.item())

    print(sum(result[-10:]))

if __name__ == '__main__':
    params = {
        'batch_size': 32,
        'lr': 0.01,
        'in_channels': 10,
        'out_channels': 10
    }
    main(params)