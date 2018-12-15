import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TaskDataset(Dataset):
    def __init__(self, params, x_path, label_path):

        self.params = params

        self.x = np.array(pd.read_csv(x_path, header=None), dtype=np.int)
        self.y = np.array(pd.read_csv(label_path, header=None), dtype=np.float32)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]