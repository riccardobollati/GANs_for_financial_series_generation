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