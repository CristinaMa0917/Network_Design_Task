import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import Test1Model, Test2Model
from dataset import TestDataset

import torcheras

def main(params):

    model = Test2Model(params['in_channels'], params['out_channels'])
    dataset = TestDataset(params, 'data/task2_data.csv', 'data/task2_label.csv')

    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    model = torcheras.Model(model, 'log/')
    model.compile(loss_fn, optimizer, metrics=['categorical_acc'])

    writer = SummaryWriter()

    def callback(epoch, i_batch, metrics_result):
        print(i_batch, metrics_result)
        for metric, value in metrics_result.items():
            writer.add_scalar('data/'+metric, value, i_batch)

    model.fit(dataloader, epochs=1, batch_callback=callback)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

if __name__ == '__main__':
    params = {
        'batch_size': 32,
        'lr': 0.01,
        'in_channels': 10,
        'out_channels': 10
    }
    main(params)