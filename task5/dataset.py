import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TaskDataset(Dataset):
    def __init__(self, params, p1_path, p2_path, label_path):

        self.params = params

        self.p1 = np.array(pd.read_csv(p1_path, header=None), dtype=np.int)
        self.p2 = np.array(pd.read_csv(p2_path, header=None), dtype=np.int)
        self.y = np.array(pd.read_csv(label_path, header=None), dtype=np.float32)

    def __getitem__(self, index):
        return (self.p1[index], self.p2[index]), self.y[index]

    def __len__(self):
        return self.p1.shape[0]