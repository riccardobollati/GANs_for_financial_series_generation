from scipy.stats import t
import statistics
import math 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

def plot_series(series, name:str, folder = None):
    
    series_to_plot = []
    for i in np.random.randint(0, len(series),size=8):
        series_to_plot.append(series[i])

    fig = plt.figure(figsize=(10, 5))
    outer = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.2)


    inner_1 = gridspec.GridSpecFromSubplotSpec(2, 4,
                    subplot_spec=outer[0], wspace=0.3, hspace=0.1)

    for ax, serie in zip(inner_1, series_to_plot):

        ax = plt.Subplot(fig, ax)
        ax.plot(serie)
        fig.add_subplot(ax)

    inner_2 = gridspec.GridSpecFromSubplotSpec(1, 1,
                    subplot_spec=outer[1], wspace=0.1, hspace=0.1)


    ax = plt.Subplot(fig, inner_2[0])
    ax.hist(series.ravel(), bins = 100)
    #ax.set_xlim(-0.25,0.25)
    ax.axvline(series.ravel().mean(), color = 'red', ls='--')
    fig.add_subplot(ax)

    if folder:
        plt.savefig(os.path.join(folder, f'{name}.png'))
    else:
        plt.show()

def samples(data_set, sample_generator, n_samples):
    '''
    this function sample n instance from the dataset and generate n
    series using the model chosen
    '''
    generated_series, it = sample_generator(n_samples)

    real_series = []
    for i in np.random.randint(0, len(data_set),size=n_samples):
        real_series.append(data_set[i].numpy())

    return np.array(real_series), np.array(generated_series), it

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
    
def save_samples_dist(real, generated, folder):
    plot_series(generated,'generated', folder)
    plot_series(real,'real', folder)


