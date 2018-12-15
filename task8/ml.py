import torch
from torch import nn
from torch import optim
from torch.utils import data
import numpy as np
import pandas as pd
import os


class Dataset(data.Dataset):
    def __init__(self,params,input_file,output_file):
        self.params = params
        self.input = pd.read_csv(input_file).values.reshape((-1,20))
        self.output = pd.read_csv(output_file).values.reshape((-1,20))

    def __getitem__(self, item):
        return self.input[item],self.output[item]

    def __len__(self):
        return self.input.shape[0]

class net_Task3(torch.nn.Module): # accuracy achieves 0.99 within 100 steps
    def __init__(self,in_num,out_num):
        super(net_Task3,self).__init__()

        self.emb = nn.Embedding(10,8) #0-9
        self.lstm = nn.LSTM(8,16,batch_first=True,dropout=1)
        self.gru = nn.GRU(16,20,batch_first=True)
        self.dense1 = nn.Linear(400,300)
        self.dense2 = nn.Linear(300,200)

    def forward(self,x):
        x = self.emb(x) # 32,20,8
        x,_ = self.lstm(x) # 32,20,16
        x,_ = self.gru(x) #32,20,10

        x = x.contiguous()
        x = x.view(x.shape[0],-1) # 32,200

        x = self.dense1(x) #32,100
        x = self.dense2(x) # 32,20

        x = x.contiguous()
        x = x.view(x.shape[0],20,10)
        return x

def accuracy(predict,output,batch):
    predict = torch.nn.functional.softmax(predict,dim=-1)
    predict = torch.max(predict,2)[1].long()
    predict.contiguous()
    predict = predict.view(batch,20)

    output.contiguous()
    output = output.view(batch,20)

    pre_num = predict.data.numpy()
    out_num = output.data.numpy()
    count = 0.0
    for i in range(batch):
        p = pre_num[i,:].tolist()
        o = out_num[i,:].tolist()
        if p==o:
            count+=1.0
    acc = count/batch
    return acc

if __name__=='__main__':
    pwd = os.getcwd()
    input_file = pwd+'/task8_train_input.csv'
    output_file = pwd+'/task8_train_output.csv'
    params = {'lr':0.02,
              'epochs':1,
              'batch_size':32,
              'in_num':10,
              'out_num':10}
    dataset = Dataset(params,input_file,output_file)
    dataloader = data.DataLoader(dataset,batch_size = params['batch_size'],shuffle=True)
    net = net_Task3(params['in_num'],params['out_num'])
    optimizer = optim.Adam(net.parameters(),lr=params['lr'])
    loss_fn = nn.CrossEntropyLoss()

    for e in range(params['epochs']):
        result = []
        for step,batch_data in enumerate(dataloader):
            net.zero_grad()
            input = batch_data[0]
            output = batch_data[1]

            predict = net(input)

            loss = 0

            batch = input.shape[0]
            for i in range(20):
                predict_i = torch.index_select(predict,1,torch.LongTensor([i]))
                predict_i.contiguous()
                predict_i = predict_i.view(batch,10).float()

                output_i = torch.index_select(output,1,torch.LongTensor([i]))
                output_i.contiguous()
                output_i = output_i.view(batch)

                # print(predict_i.shape)
                # print(output_i.shape)
                loss += loss_fn(predict_i,output_i)
            # loss = loss_fn(predict,output)
            loss.backward()


            optimizer.step() # apply gradients

            temp_r = accuracy(predict,output,batch)
            result.append(temp_r)
            print('Epoch [ %d]  step: %d Accuracy : %s'%(e,step,temp_r))

    print('final 100 step mean accuracy:',np.mean(result[-100:]))



