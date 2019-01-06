import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pandas as pd
import numpy as np

from model import TaskModel
from dataset import TaskDataset

import torcheras

def main(params):

    device = torch.device('cuda:0')

    model = TaskModel(params)
    train_dataset = TaskDataset(params, 'data/blob_train_image_data/', 'data/train_sym.txt')
    # test_dataset = train_dataset[int(len(train_dataset)*0.8):]
    # train_dataset = train_dataset[:int(len(train_dataset)*0.8)]

    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params['lr'])

    model = torcheras.Model(model, 'log/')
    model.compile(loss_fn, optimizer, metrics=['binary_acc'], device=device)

    writer = SummaryWriter()

    result = []
    def callback(epoch, i_batch, metrics_result):
        result.append(metrics_result['binary_acc'])

    model.fit(train_dataloader, epochs=10, batch_callback=callback)
    # print(sum(result[-100:]))

    # p1 = np.array(pd.read_csv('data/task5_2_test_paragraph1.csv', header=None), dtype=np.int)
    # p2 = np.array(pd.read_csv('data/task5_2_test_paragraph2.csv', header=None), dtype=np.int)

    # results = []
    # for i in range(p1.shape[0]):
    #     result = model.model.forward((torch.LongTensor([p1[i]]), torch.LongTensor([p2[i]])))
    #     if result > 0.5:
    #         results.append(1)
    #     else: 
    #         results.append(0)
    # print(results)

if __name__ == '__main__':
    params = {
        'in': 30,
        'embed_dim': 8,
        'batch_size': 32,
        'lr': 0.001,
        'out_dim':2
    }
    main(params)
