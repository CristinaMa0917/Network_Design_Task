import torch
from torch import nn
from torch import optim
from torch.utils import data
import numpy as np
import pandas as pd
import os
import pandas as pd
from PIL import Image

class Dataset(data.Dataset):
    def __init__(self,params,data,cls,sym):
        self.params = params
        self.cls = pd.read_table(cls,header=None,delim_whitespace=True).values[:,1] # 1~20000
        self.sym = pd.read_table(sym,header=None,delim_whitespace=True).values[:,1] # 1~20000
        self.images = []
        for i in range(len(self.cls)):
            im = Image.open(data+'blob_train_image_data%d.png'%(i+1))
            im = np.array(im)
            self.images.append(im)
        self.images = np.array(self.images)

    def __getitem__(self, item):
        return self.images[item],self.cls[item],self.sym[item]

    def __len__(self):
        return self.images.shape[0]

class net_Task7_cls(torch.nn.Module): # accuracy achieves 0.99 within 100 steps
    def __init__(self,in_num,out_num):
        super(net_Task7_cls,self).__init__()

        self.conv1 = nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1)
        self.pool1 = nn.MaxPool2d(4,stride=2,padding=1)

        self.conv2 = nn.Conv2d(4,16,3,1,1)
        self.pool2 = nn.MaxPool2d(4,2,1)

        self.conv3 = nn.Conv2d(16,32,3,1,1)
        self.pool3 = nn.MaxPool2d(4,2,1)

        self.conv4 = nn.Conv2d(32,64,3,1,1)
        self.pool4 = nn.MaxPool2d(4,2,1)

        self.dense1 = nn.Linear(2*2*64,64)
        self.dense2 = nn.Linear(64,16)
        self.dense3 = nn.Linear(16,2)

    def forward(self,image):
        image = image.unsqueeze(-1)
        x = image.permute(0,3,1,2)

        x = self.conv1(x)
        x = self.pool1(x)
        x = nn.functional.relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = nn.functional.relu(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = nn.functional.relu(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = nn.functional.relu(x)

        x = x.contiguous()
        x = x.view(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.nn.functional.softmax(x)

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
    data_file = pwd+'/data/blob_train_image_data/'
    cls_file = pwd+'/data/train_cls.txt'
    sym_file = pwd+'/data/train_sym.txt'
    params = {'lr':0.01,
              'epochs':1,
              'batch_size':32,
              'in_num':10,
              'out_num':10}
    dataset = Dataset(params,data_file,cls_file,sym_file)
    dataloader = data.DataLoader(dataset,batch_size = params['batch_size'],shuffle=True)
    net = net_Task7_cls(params['in_num'],params['out_num'])
    optimizer = optim.Adam(net.parameters(),lr=params['lr'])
    loss_fn = nn.CrossEntropyLoss()

    for e in range(params['epochs']):
        result = []
        for step,batch in enumerate(dataloader):
            net.zero_grad()
            img = batch[0].long()
            cls = batch[1].long()
            sym = batch[2].long()

            output = net(img)
            loss = loss_fn(output,cls)
            loss.backward()

            optimizer.step() # apply gradients

            temp_r = accuracy(output,cls)
            result.append(temp_r)
            print('Epoch [ %d]  step: %d Accuracy : %s'%(e,step,temp_r))

    print('final 100 step mean accuracy:',np.mean(result[-100:]))

    # query_file_t = pwd+'/data/task3_query.csv'
    # passa_file_t = pwd+'/data/task3_query.csv'
    # label_file_t = pwd+'/data/task3_query.csv'
    # dataset = Dataset(params, query_file, passa_file, label_file)
    # dataloader = data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)




