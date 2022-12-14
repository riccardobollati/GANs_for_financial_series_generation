import os
import shutil
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from models import Generator
import tkinter as tk
from tkinter import filedialog
import random


def create_dir(location, name):
    os.mkdir(os.path.join(location, name))
    shutil.copy('models.py', os.path.join(location, name))
    os.mkdir(os.path.join(location, name, 'training_samples'))
    os.mkdir(os.path.join(location, name, 'checkpoints'))
    os.mkdir(os.path.join(location, name, 'training_samples','dynamic_latents'))
    os.mkdir(os.path.join(location, name, 'training_samples','fixed_latents'))
    return os.path.join(location, name)

# the following function allow to generate a sample to test the model
def generate_sample(model, n_samples = 1, tensor = True):
    samples = []
    for i in range(n_samples):
        pr = model(torch.rand((1,50)))
        if tensor:
            samples.append(pr)
        else:
            samples.append(pr.detach().numpy())
    return samples

def regression(prices, only_resid = False):

    linear_reg = LinearRegression(fit_intercept=True)
    linear_reg.fit(np.linspace(1,100, num=100).reshape(-1,1), prices)
    prediction = linear_reg.predict(np.linspace(1,100, num=100).reshape(-1,1))
    residuals = (prices - prediction)
    coef = linear_reg.coef_

    if only_resid:
        return residuals
    else:
        return prediction, residuals, coef

def load_model():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    print(f'the model has been selected: {file_path}')

    #instantiate the model
    g = Generator()
    g.load_state_dict(torch.load(file_path)['g_state_dict'])

    return g

class SamplesPlot:

    
    def plot_samples_price(sample, return_series = False, save = None):

        fig, axs = plt.subplots(len(sample), 3)
        fig.set_facecolor('white')
        fig.set_figheight(3.5*len(sample))
        fig.set_figwidth(12)

        prices_list = []
        residuals_list = []
        coeff_list = []

        for ax, samp in zip(axs, sample):

            ax[0].plot(samp)
            ax[0].axhline(0, ls ='--', color = 'black')

            round_mean = str(np.round(samp.mean(),4))
            round_sd   = str(np.round(samp.std(),4))

            ax[0].set_title(f'mean: {round_mean} | sd: {round_sd}', fontsize=8)
            
            generated_price = []
            
            init = 1
            for p in samp:
                init = init + init * p.item()
                generated_price.append(init)
            
            ax[1].plot(generated_price)
            ax[1].set_title('Prices', fontsize=8)
            
            prediction, residuals, coef = regression(generated_price)

            prices_list.append(generated_price)
            residuals_list.append(residuals)
            coeff_list.append(coef)

            ax[1].plot(prediction, color = 'red', alpha = 0.6)

            ax[2].scatter(np.linspace(1,100, num=100),residuals)
            ax[2].axhline(0, color = 'black', ls ='--')
            ax[2].set_title(f'var: {np.round(np.var(residuals),4)}')

        if save:
            plt.savefig(save)

        if return_series:
            return prices_list, residuals_list, coeff_list
    

    def real_fake_comparison(plain_dataset,syntetic_df, sample_size:list, save=False):

        real = [plain_dataset[i].detach().numpy()/plain_dataset[i].detach().numpy()[0] for i in np.random.randint(0, len(plain_dataset),size=sample_size[0])]
        fake = [syntetic_df[i] for i in np.random.randint(0, 10,size=sample_size[1])]

        generated_price = []

        for i in fake:
            single = []
            init = 1
            for p in i:
                init = init + init * p.item()
                single.append(init)

            generated_price.append(single)


        series = real + generated_price

        label = ['real']*sample_size[0] +['fake']*sample_size[1]

        toplot = list(zip(series, label))

        random.shuffle(toplot)

        series, labels = zip(*toplot)

        rows = (sample_size[0]+sample_size[1]) / 5

        fig, axs = plt.subplots(rows,5)
        fig.set_figheight(6)
        fig.set_figwidth(20)

        for ax, serie, label in zip(axs.ravel(),series, labels):
            ax.plot(serie)
            ax.axhline(1, color='black', ls='--', alpha = 0.6)
            if label == 'fake':
                ax.set_facecolor('gainsboro')
        
        plt.show()
        if save:
            plt.savefig('results/comparison_README.png')
