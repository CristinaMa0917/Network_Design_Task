import torch
from torch import nn
from torch import optim
from torch.utils import data
import numpy as np
import pandas as pd
import os


class Dataset(data.Dataset):
    def __init__(self,params,query_file,passa_file,label_file):
        self.params = params
        self.query = pd.read_csv(query_file).values
        self.passa = pd.read_csv(passa_file).values
        self.label = pd.read_csv(label_file).values.reshape([-1])
        # self.oh_label = np.zeros((len(self.label),10))
        # self.oh_label[:,self.label]=1

    def __getitem__(self, item):
        return self.query[item],self.passa[item],self.label[item]

    def __len__(self):
        return self.query.shape[0]

class net_Task3(torch.nn.Module): # accuracy achieves 0.99 within 100 steps
    def __init__(self,in_num,out_num):
        super(net_Task3,self).__init__()
        self.emb = nn.Embedding(11,8)
        self.lstm = nn.LSTM(16,10,batch_first=True,dropout=1)
        self.gru = nn.GRU(10,10,batch_first=True)
        self.dense1 = nn.Linear(100,20)
        self.dense2 = nn.Linear(20,out_num)

    def forward(self,query,passage):
        query_emb = self.emb(query) # 32,1,8
        passa_emb = self.emb(passage) # 32,10,8
        query_emb = query_emb.repeat(1,10,1)

        x = torch.cat([query_emb,passa_emb],2) # 32,10,16
        x,_ = self.lstm(x) # 32,11,10
        x,_ = self.gru(x) #32,11,15

        x = x.contiguous()
        x = x.view(x.shape[0],-1)

        x = self.dense1(x) #
        x = self.dense2(x)
        # x = self.softmax(x)
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
    query_file = pwd+'/data/task3_query.csv'
    passa_file = pwd+'/data/task3_passage.csv'
    label_file = pwd+'/data/task3_label.csv'
    params = {'lr':0.01,
              'epochs':1,
              'batch_size':32,
              'in_num':10,
              'out_num':10}
    dataset = Dataset(params,query_file,passa_file,label_file)
    dataloader = data.DataLoader(dataset,batch_size = params['batch_size'],shuffle=True)
    net = net_Task3(params['in_num'],params['out_num'])
    optimizer = optim.Adam(net.parameters(),lr=params['lr'])
    loss_fn = nn.CrossEntropyLoss()

    for e in range(params['epochs']):
        result = []
        for step,batch in enumerate(dataloader):
            net.zero_grad()
            b_q = batch[0]
            b_p = batch[1]
            b_y = batch[2].long()

            output = net(b_q,b_p)
            loss = loss_fn(output,b_y)
            loss.backward()


            optimizer.step() # apply gradients

            temp_r = accuracy(output,b_y)
            result.append(temp_r)
            print('Epoch [ %d]  step: %d Accuracy : %s'%(e,step,temp_r))

    print('final 100 step mean accuracy:',np.mean(result[-100:]))

    # query_file_t = pwd+'/data/task3_query.csv'
    # passa_file_t = pwd+'/data/task3_query.csv'
    # label_file_t = pwd+'/data/task3_query.csv'
    # dataset = Dataset(params, query_file, passa_file, label_file)
    # dataloader = data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)




