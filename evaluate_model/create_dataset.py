from torch.utils.data import Dataset
import os
import torch
import numpy as np

class WindowsDaset(Dataset):
    
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir

        self.transform = transform
    
    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        if type(idx) == list:
            response = []
            for i in idx:
                path = os.path.join(self.data_dir, os.listdir(self.data_dir)[i])
                series = torch.tensor(np.genfromtxt(path, delimiter=',')[1:])
                response.append(series)
            
            if self.transform:
                response = [self.transform(i) for i in response]

            return response
        else:
            path = os.path.join(self.data_dir, os.listdir(self.data_dir)[idx])
            series = torch.tensor(np.genfromtxt(path, delimiter=',')[1:])

            if self.transform:
                series = self.transform(series)
                
            return series