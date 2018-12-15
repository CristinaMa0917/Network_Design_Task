
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TaskDataset(Dataset):
    def __init__(self, d1, d2, d3, d4):

        self.x = np.array(np.random.random_integers(0, 1, (320000, 4)), dtype=np.float32)
        self.y = np.array(self.x.sum(1) % 2, dtype=np.int)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]