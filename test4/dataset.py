import torch
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, params, x_dir, x_query_dir, y_dir):

        self.params = params

        self.x = np.array(pd.read_csv(x_dir), dtype=np.int)
        self.x_query = np.array(pd.read_csv(x_query_dir), dtype=np.int)
        self.y = np.array(pd.read_csv(y_dir), dtype=np.int).squeeze(-1)

    def __getitem__(self, index):
        return (self.x[index], self.x_query[index], self.y[index]), self.y[index]

    def __len__(self):
        return self.x.shape[0]