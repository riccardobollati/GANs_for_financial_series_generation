import os
import shutil
import torch
import matplotlib.pyplot as plt
import numpy as np

def create_dir(location, name):
    os.mkdir(os.path.join(location, name))
    shutil.copy('models.py', os.path.join(location, name))
    os.mkdir(os.path.join(location, name, 'training_samples'))
    os.mkdir(os.path.join(location, name, 'checkpoints'))
    os.mkdir(os.path.join(location, name, 'training_samples','dynamic_latents'))
    os.mkdir(os.path.join(location, name, 'training_samples','fixed_latents'))
    return os.path.join(location, name)

# the following function allow to generate a sample to test the model
def generate_sample(model, n_samples = 1):
    samples = []
    for i in range(n_samples):
        pr = model(torch.rand((1,50)))
        samples.append(pr)
    return samples

def plot_samples_price(sample):

    fig, axs = plt.subplots(len(sample), 2)
    fig.set_facecolor('white')
    fig.set_figheight(17)
    fig.set_figwidth(12)

    prices = []

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
        
        prices.append(generated_price)
    
    return prices
