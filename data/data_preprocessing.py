import pandas as pd
import numpy as np
import yfinance as yf
import os
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

def get_data(tiker_list, destination):
    
    df = pd.read_csv(tiker_list)
    for i in df['Holding Ticker']:
        i = i.replace(' ','')
        print(f'downloading: {i}')
        series = yf.download(i, progress=False)
        series.reset_index(drop=True,inplace=True)
        series.iloc[:,4].to_csv(f'{destination}\{i}.csv',index = False)

def get_next_window(series, start, window_len):
    if (start + window_len) > len(series):
        return False
    else:
        return series[start:(start+window_len+1)]

def get_windows(df, window_len):
    windows = []
    i = 0
    while True:
        window = get_next_window(df, i, window_len)
        if type(window) == np.ndarray:
            windows.append(window)
        else:
            break
        i += window_len
    return windows

def get_returns(file):
    df = pd.read_csv(f'series/{file}')
    df = np.array(df['Adj Close'])
    return np.diff(df,1)/df[:-1]

def create_windows_df(folder, wind_size):
    os.mkdir(f'dataset_{wind_size}_winds')

    for file in tqdm(os.listdir(folder)):
        returns = get_returns(file)
        winds = get_windows(returns,wind_size)
        save_single_windows(winds, file.split('.')[0], f'dataset_{wind_size}_winds')

def save_single_windows(winds, tiker, folder):
    
    for index, w in enumerate(winds):
        np.savetxt(f'{folder}\{tiker}_{index}.csv', w, delimiter=',')
       
    

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