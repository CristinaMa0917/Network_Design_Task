import torch
from torch import nn
from torch import optim
from torch.utils import data
import numpy as np
import pandas as pd
import os


class Dataset(data.Dataset):
    def __init__(self,params,data_file,label_file):
        self.params = params
        self.data = pd.read_csv(data_file).values
        self.label = pd.read_csv(label_file).values.reshape([-1])
        # self.oh_label = np.zeros((len(self.label),10))
        # self.oh_label[:,self.label]=1

    def __getitem__(self, item):
        return self.data[item],self.label[item]

    def __len__(self):
        return self.data.shape[0]

class net_Task1(torch.nn.Module): # accuracy achieves 0.99 within 100 steps
    def __init__(self,in_num,out_num):
        super(net_Task1,self).__init__()

        self.dense1 = nn.Linear(in_num,20)
        self.dense2 = nn.Linear(20,out_num)
        # self.softmax = nn.functional.softmax()

    def forward(self,x):
        x = x.float()
        x = self.dense1(x)
        x = self.dense2(x)
        # x = self.softmax(x)
        return x

class net_Task2(torch.nn.Module):
    def __init__(self,in_num,out_num):
        super(net_Task2,self).__init__()

        self.conv = nn.Conv1d(1,1,kernel_size=3,padding=1)
        self.dense1 = nn.Linear(10,30)
        self.dense2 = nn.Linear(30,10)

    def forward(self,x):
        x = x.float()
        x = x.unsqueeze(-1) # 32,10,1
        x = x.permute(0,2,1) # 32,1 10
        x = self.conv(x) # 32,1,8
        x = x.squeeze(1) #32,8
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def accuracy(output,labels):
    labels = labels.contiguous()

    output = torch.nn.functional.softmax(output)
    output = torch.max(output,1)[1].long()
    eqs = labels.eq(output).float()
   # print(eqs)
    acc = torch.mean(eqs,0).float()
    acc = acc.numpy()
    return acc

if __name__=='__main__':
    pwd = os.getcwd()
    data_file = pwd+'/data/task2_data.csv'
    label_file = pwd+'/data/task2_label.csv'
    params = {'lr':0.01,
              'epochs':1,
              'batch_size':32,
              'in_num':10,
              'out_num':10}
    dataset = Dataset(params,data_file,label_file)
    dataloader = data.DataLoader(dataset,batch_size = params['batch_size'],shuffle=True)
    net = net_Task2(params['in_num'],params['out_num'])
    optimizer = optim.Adam(net.parameters(),lr=params['lr'])
    loss_fn = nn.CrossEntropyLoss()

    for e in range(params['epochs']):
        result = []
        for step,batch in enumerate(dataloader):
            net.zero_grad()
            b_x = batch[0]
            b_y = batch[1].long()

            output = net(b_x)
            loss = loss_fn(output,b_y)
            loss.backward()


            optimizer.step() # apply gradients

            temp_r = accuracy(output,b_y)
            result.append(temp_r)
            print('Epoch [ %d]  step: %d Accuracy : %s'%(e,step,temp_r))

    print('final 100 step mean accuracy:',np.mean(result[-100:]))





