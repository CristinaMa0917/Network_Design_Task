import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import Task3Model
from dataset import TestDataset

import torcheras

def loss_fn(y_pred, y_true):
    output = y_pred
    # create inverted indices
    idx = [i for i in range(output.shape[0]-1, -1, -1)]
    idx = torch.LongTensor(idx)
    inverted_output = output.index_select(0, idx)
    output_distances = F.cosine_similarity(output, inverted_output, 1, 1e-8)
    
    inverted_y_true = y_true.index_select(0, idx)
    y = (y_true == inverted_y_true).float()
    result = (output_distances * y).sum()
    return result

def main(params):

    model = Task3Model()
    dataset = TestDataset(params, 'data/task4_train_passage.csv', 'data/task4_train_query.csv', 'data/task4_train_label.csv')

    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.TripletMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    model = torcheras.Model(model, 'log/')
    # model.compile(loss_fn, optimizer, metrics=['categorical_acc'])
    model.compile(loss_fn, optimizer)

    writer = SummaryWriter()

    result = []
    def callback(epoch, i_batch, metrics_result):
        # print(i_batch, metrics_result)
        for metric, value in metrics_result.items():
            writer.add_scalar('data/'+metric, value, i_batch)
        # result.append(metrics_result['categorical_acc'])

    model.fit(dataloader, epochs=1, batch_callback=callback)

    # writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    print(sum(result[-100:]))

if __name__ == '__main__':
    params = {
        'batch_size': 32,
        'lr': 0.01,
        'in_channels': 10,
        'out_channels': 10
    }
    main(params)