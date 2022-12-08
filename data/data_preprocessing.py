import pandas as pd
import numpy as np
import yfinance as yf
import os
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

def get_data(tiker_list, destination):
    '''
    this function download all the stocks time series. 
    The stocks downloaded are contained in the "value_st.csv" file.
    '''   
    os.mkdir(destination) 

    df = pd.read_csv(tiker_list)

    for i in (pbar := tqdm(df['Holding Ticker'])):
        i = i.replace(' ','')
        pbar.set_description(f'downloading: {i}')
        #print(f'downloading: {i}',flush=True)
        series = yf.download(i, progress=False)
        series.reset_index(drop=True,inplace=True)
        series.iloc[:,4].to_csv(f'{destination}\{i}.csv',index = False)

def get_next_window(series, start, window_len):
    '''
    this function return the next windows given an index.
    ::param series: the series used to generate windows
    ::param start: the starting index of the window
    ::param window_len: the lenght of the windows we want to generate
    '''
    if (start + window_len) > len(series):
        return False
    else:
        return series[start:(start+window_len+1)]

def get_windows(df, window_len):
    '''
    given a dataset this function divides it into smaller windows and return them
    as a list
    ::param df: source dataset
    ::param window_len: the lenght of the windows we want to extract
    '''
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

def get_data_from_folder(folder, file, returns = False):
    '''
    return the returns from the prices
    ::param file: the source file that contains all the prices 
    '''
    df = pd.read_csv(f'{folder}/{file}')
    df = np.array(df['Adj Close'])
    if returns:
        return np.diff(df,1)/df[:-1]
    else:
        return df

def create_windows_df(folder, wind_size, destination, get_returns = False):
    '''
    create the dataset wich contains all the windows extracted from the original series
    ::param folder: source folders wich contains all the time series
    ::param wind_size: lenaght of the windows we want to generate
    '''
    os.mkdir(destination)

    for file in (pbar := tqdm(os.listdir(folder))):
        pbar.set_description(f"processing {file.split('.')[0]}")
        data = get_data_from_folder(folder,file, returns=get_returns)
        winds = get_windows(data,wind_size)
        save_single_windows(winds, file.split('.')[0], destination)

def save_single_windows(winds, tiker, folder):
    '''
    save the stock windows
    ::param winds: the extracted windows
    ::param ticker: the name of the stock
    ::param folder: destination folder
    '''
    
    for index, w in enumerate(winds):
        np.savetxt(os.path.join(folder,f'{tiker}_{index}.csv'), w, delimiter=',')
       
    

class WindowsDaset(Dataset):
    
    def __init__(self, data_dir, transform = None, data = 'returns'):
        self.data_dir = data_dir
        self.transform = transform
        self.data = data
    
    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        
        if self.data == 'returns':
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
        else:
            if type(idx) == list:
                response = []
                for i in idx:
                    path = os.path.join(self.data_dir, os.listdir(self.data_dir)[i])
                    series = torch.tensor(np.genfromtxt(path, delimiter=',')[1:])

                    generated_price = []
        
                    init = 1

                    for p in series:
                        init = init + init * p.item()
                        generated_price.append(init)
                    
                    response.append(generated_price)

                return response
            else:
                path = os.path.join(self.data_dir, os.listdir(self.data_dir)[idx])
                series = torch.tensor(np.genfromtxt(path, delimiter=',')[1:])

                generated_price = []
        
                init = 1
                for p in series:
                    init = init + init * p.item()
                    generated_price.append(init)
                    
                return generated_price