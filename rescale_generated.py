import numpy as np
import torch
from tqdm import tqdm

class Rescale:
    
    def __init__(self, data_set, picks_range) -> None:
        
        print('~Scaler initialization~')
        self.data_set = data_set
        self.len = len(data_set)
        self.picks_range = picks_range
        
        print('~computing the min and max quantiles~')
        maxes = np.zeros(len(data_set))
        mins  = np.zeros(len(data_set))
        for index, i in tqdm(enumerate(data_set), total=self.len):
            maxes[index] = i.max().item()
            mins[index]  = i.min().item()

        self.q_max   = np.quantile(maxes,[0.1 * i for i in range(1,10)])
        self.q_min   = np.quantile(abs(mins),[0.1 * i for i in range(1,10)])*-1

        print('~computing the distance between peaks~')
        distances_min = np.zeros(len(data_set))
        distances_max = np.zeros(len(data_set))

        for index, i in tqdm(enumerate(data_set), total=self.len):
            max_2, max_1 = (i.sort()[0][-2:])
            distances_max[index] = max_1 - max_2

            min_1, min_2 = (i.sort()[0][:2])
            distances_min[index] = abs(min_2) - abs(min_1)
            

        self.dist_q_max = np.quantile(distances_max,[0.1 * i for i in range(1,10)])
        self.dist_q_min = np.quantile(distances_min,[0.1 * i for i in range(1,10)])
    
    def scale(self, sample):
        
        #scaler parameters
        q_dist_max = self.dist_q_max
        q_dist_min = self.dist_q_min 
        q_max      = self.q_max
        q_min      = self.q_min

        max_2, max_1 = (sample.sort()[0][-2:])
        dist = max_1 - max_2
        dist = (dist/(abs(sample.min())+sample.max())).item()

        sorted_q   = np.sort(np.append(q_dist_max,dist))
        dist_index =  np.where(sorted_q == dist)[0]
        if dist_index == 9:
            low_rnd_max = q_max[8]
            up_rnd_max  = q_max[8]+ self.picks_range
        elif dist_index == 0:
            up_rnd_max  = q_max[0]
            low_rnd_max = q_max[0] -0.01
        else:
            low_rnd_max = q_max[dist_index-1]
            up_rnd_max  = q_max[dist_index]
        
        max_pick = np.random.uniform(low=low_rnd_max, high=up_rnd_max)

        #pick value for min
        min_1, min_2 = (sample.sort()[0][:2])
        dist_min = abs(min_2) - abs(min_1)
        dist_min = (dist_min/(abs(sample.min())+sample.max())).item()

        sorted_q_min   = np.sort(np.append(q_dist_min,dist_min))
        dist_index_min =  np.where(sorted_q_min == dist_min)[0]

        if dist_index_min == 9:
            low_rnd_min = q_min[8]
            up_rnd_min  = q_min[8]+0.01
        elif dist_index_min == 0:
            up_rnd_min  = q_min[0]
            low_rnd_min = q_min[0] - self.picks_range
        else:
            low_rnd_min = q_min[dist_index_min-1]
            up_rnd_min = q_min[dist_index_min]
        min_pick = np.random.uniform(low=low_rnd_min, high=up_rnd_min)


        max_sample = sample.max().item()
        min_sample = sample.min().item()
        sample = sample.detach().numpy()
        X_std = (sample - min_sample) / (max_sample - min_sample)
        X_scaled = X_std * (max_pick - min_pick) + min_pick

        return X_scaled
