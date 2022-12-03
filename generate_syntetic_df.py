import numpy as np
import torch
from utils import regression
from sklearn.preprocessing import MinMaxScaler
from rescale_generated import Rescale
from evaluate_model.create_dataset import WindowsDaset


class DfGenerator:

    def __init__(self, variance_th:list, mean_boundary:float, max_range:float, generator, data_set = None, rescaler = None):
        
        self.variance_th = variance_th
        self.generator   = generator
        self.mean_bond   = mean_boundary
        self.max_range   = max_range

        if rescaler:
            self.rescaler = rescaler
        else:
            dataset = WindowsDaset(data_set)
            self.rescaler = Rescale(dataset)

        
    def __call__(self,size):

        syntetic_df = []
        diff        = []

        it = 0
        while len(syntetic_df) < size:
            
            print(f'found: {len(syntetic_df)}/{size} over {it} iterations', end='\r')
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
            

            if cumulative_difference >= self.variance_th[0] and cumulative_difference <= self.variance_th[1]:
                syntetic_df.append(scaled)
                diff.append(cumulative_difference)
        
            it += 1
        
        return syntetic_df, it



