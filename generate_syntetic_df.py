import numpy as np
import torch
from utils import regression
from sklearn.preprocessing import MinMaxScaler
from rescale_generated import Rescale
import data.data_preprocessing as data_prep

class DfGenerator:

    def __init__(self, variance_th, mean_boundary, max_range, generator, size, data_set = None, rescaler = None) -> None:
        
        self.variance_th = variance_th
        self.generator   = generator
        self.size        = size
        self.mean_bond   = mean_boundary
        self.max_range   = max_range

        if rescaler:
            self.rescaler = rescaler
        else:
            dataset = data_prep.WindowsDaset(data_set)
            self.rescaler = Rescale(dataset)

        

    def __call__(self):

        syntetic_df = []
        diff        = []

        it = 0
        while len(syntetic_df) < self.size:
            
            print(f'iteration N: {it}', end='\r')
            generated = self.generator(torch.rand((1,50)))
            scaled = self.rescaler.scale(generated.flatten())

            generated_price = []
            init = 1
            for p in scaled:
                init = init + init * p.item()
                generated_price.append(init)
            
            if (abs(np.mean(scaled)) > self.mean_bond) or (generated_price[-1] >= 1 + self.max_range) or (generated_price[-1] <= 1 - self.max_range):
                it += 1
                continue
            
            resid = regression(generated_price, only_resid=True)
            scaler = MinMaxScaler()
            resid = scaler.fit_transform(resid.reshape(-1, 1))
            
            cumulative_difference = 0
            
            for i in range(len(resid)-1):
                cumulative_difference += abs(abs(resid[i]) - abs(resid[i+1]))
            

            if cumulative_difference >= self.variance_th:
                syntetic_df.append(scaled)
                diff.append(cumulative_difference)
        
            it += 1
        
        return syntetic_df, diff



