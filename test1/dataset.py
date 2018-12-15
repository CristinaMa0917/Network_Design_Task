import torch
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, params, x_dir, y_dir):

        self.params = params

        self.x = np.array(pd.read_csv(x_dir), dtype=np.int)
        self.y = np.array(pd.read_csv(y_dir), dtype=np.int).squeeze(-1)

    def __getitem__(self, index):
        # return self.x[index], self.to_onehot(self.y[index])
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]