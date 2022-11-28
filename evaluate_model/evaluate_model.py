#from models import Generator
import torch
import tkinter as tk
from tkinter import filedialog
from models import Generator
import matplotlib.pyplot as plt
from scipy.stats import t
import statistics
import math

from create_dataset import WindowsDaset
import numpy as np
import plot_comparison

#select the model to use
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
print(f'the model has been selected: {file_path}')

#define the number of instance to sample
number_of_samples = int(input('number of samples: '))

#instantiate the model
g = Generator()
g.load_state_dict(torch.load(file_path)['g_state_dict'])

#real data dataset
dataset = WindowsDaset(r'C:\Users\bolla\Desktop\CUHK_courses\IASP_elisa\final\data\dataset_100_winds')

def samples(model, n_samples):
    '''
    this function sample n instance from the dataset and generate n
    series using the model chosen
    '''
    generated_series = []
    for i in range(n_samples):
        serie = model(torch.rand((1,50)))/1000
        generated_series.append(serie.detach().numpy().ravel())

    
    real_series = []
    for i in np.random.randint(0, len(dataset),size=n_samples):
        real_series.append(dataset[i].numpy())

    return np.array(real_series), np.array(generated_series)

#test media = 0
def mean_average(real_serie, generated_serie, alpha):
    '''
    this function test if the generated sample mean is equal to
    the real data sample
    '''
    n    = len(real_serie)
    df   = n-1

    mu_0 = statistics.mean(real_serie)

    gen_mean = statistics.mean(generated_serie)
    gen_var  = statistics.variance(generated_serie)
    t_alpha = t.ppf(alpha, df = df)

    h_alpha_upper = mu_0 + t_alpha * (math.sqrt(gen_var)/math.sqrt(n))
    h_alpha_lower = mu_0 - t_alpha * (math.sqrt(gen_var)/math.sqrt(n))

    print(f'rejection theshold: {np.round(h_alpha_upper,5)} - {np.round(h_alpha_lower,5)}')
    print(f'sample mean: {gen_mean}')

    stat_test = (gen_mean-mu_0/math.sqrt(gen_var/n))
    if (stat_test >= h_alpha_upper) or (stat_test <= h_alpha_lower):
        print('H0 rejected')
    else:
        print('H0 accepted')


real_series, generated_series = samples(g, number_of_samples)

plot_comparison.plot_series(generated_series)
mean_average(real_series.ravel(), generated_series.ravel(),0.1)





