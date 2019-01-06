import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from PIL import Image

class TaskDataset(Dataset):
    def __init__(self, params, x_path, label_path):
        self.x_path = x_path
        self.x = []
        self.y = []
        # print(label_path)
        with open(label_path, 'r') as f:
            for line in f.readlines():
                # print(line)
                img_path, label = line.split()
                self.x.append(img_path)
                self.y.append(int(label))
        
        self.y = np.array(self.y, dtype=np.float32)

    def __getitem__(self, index):
        img = np.array(Image.open(self.x_path+self.x[index]), dtype=np.float32) / 255.
        label = self.y[index]
        return img, label

    def __len__(self):
        return len(self.x)