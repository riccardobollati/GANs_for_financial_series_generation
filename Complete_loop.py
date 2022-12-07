import tkinter as tk
from tkinter import filedialog
from models import Generator
import torch
from evaluate_model.create_dataset import WindowsDaset
from rescale_generated import Rescale
import utils
import matplotlib.pyplot as plt
import os
from generate_syntetic_df import DfGenerator
import argparse
from evaluate_model import evaluate_model
from train import Trainer
from torch.utils.data import DataLoader


#select the model to use
def chose_model():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    print(f'the model has been selected: {file_path}')
    
    return file_path

def init_model(file_path):
    g = Generator()
    g.load_state_dict(torch.load(file_path)['g_state_dict'])
    return g

def select_dataset(transform = None):
    dataset_folder = filedialog.askdirectory()
    dataset = WindowsDaset(dataset_folder, transform)
    
    return dataset, dataset_folder

def plot_generated_r(generator, n_samples, scaler, folder):
    sample = utils.generate_sample(generator,n_samples)
    scaled = []
    #scale the generated series
    for i in sample:
        scaled.append(scaler.scale(i.flatten()))

    fig, axs = plt.subplots(len(sample),2)
    fig.set_figheight(3*len(sample))
    fig.set_figwidth(10)

    for i,ax in enumerate(axs):
        sample[i] = sample[i].detach().numpy()
        ax[0].plot(sample[i][0])
        ax[1].plot(scaled[i])
        ax[1].axhline(0, color='black', ls='--', alpha = 0.6)
        ax[1].set_facecolor('gainsboro')
    
    plt.plot()
    plt.savefig(os.path.join(folder, 'generated returns.png'))

    return sample, scaled

class ScaleInput:
    
    def __init__(self, scale) -> None:
        self.scale = scale

    def __call__(self, sample):
        return sample * self.scale

if __name__ == '__main__':

    Train_ = bool(int(input('input 1 for training [0] for not training ')) or 0)
    #results destination folder for generated img:
    input('select a destination folder for plots and model...')
    destination_f = filedialog.askdirectory()

    #load dataset and initialize rescaler
    print('select the dataset to use...')
    dataset, dataset_folder = select_dataset()
    picks_range = float(input('select a range for the highest/lowest peacks: '))

    

    if Train_:
        from models import Critic
        g = Generator()
        g_opt = torch.optim.RMSprop(g.parameters(), lr=0.1)

        c = Critic()
        c_opt = torch.optim.RMSprop(c.parameters(), lr=0.05)
        
        
        #chose the Base directiory to save the run
        BASE_DIR = destination_f

        #global variables for the run
        GP_WEIGHT = 10
        G_NORM_PEN = 5
        CRITIC_IT = 5
        PRINT_EVERY = 10
        CHECKPOINT_FREQ = 10
        BATCH_SIZE = 256
        EPOCHS = 1
        run_ARCHIVE = utils.create_dir(BASE_DIR,'model_training')

        #create dataset and transformer
        ds_transformer = ScaleInput(100)
        #create the dataloader
        train_dataset = WindowsDaset(dataset_folder, transform = ds_transformer)
        data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)


        # Instantiate Trainer
        trainer = Trainer(g, c, g_opt, c_opt,GP_WEIGHT,G_NORM_PEN,CRITIC_IT,PRINT_EVERY,CHECKPOINT_FREQ, ARCHIVE_DIR=run_ARCHIVE)
        # Train model
        print('Training is about to start...')

        trainer.train(data_loader, epochs=EPOCHS, plot_training_samples=True, checkpoint=None)
    else:
        #load model
        print('chose the model to use...')
        model_params_path = chose_model()
        print('loading the model...')
        g = init_model(model_params_path)
    
    #n_samples to show:
    n_samples = int(input('decide how many samples to show: '))

    print('initializing the Rescaler...')
    rescaler = Rescale(dataset, picks_range=picks_range)

    #plot generated series
    sample, scaled = plot_generated_r(generator=g,
                    n_samples=n_samples,
                    scaler=rescaler,
                    folder=destination_f)
    
    #generate syntetic series
    syn_num = int(input('select the number of syntetic series to generate for the plot: '))
    df_generator = DfGenerator(variance_th=[6,8], max_range=0.15, generator=g, rescaler=rescaler)
    syntetic_df, it = df_generator(syn_num)

    utils.plot_samples_price(syntetic_df, save=os.path.join(destination_f, 'syntetic prices.png'), return_series=False)

    #evaluate disctribution
    eval_sample = int(input('select the dimension of the sample to evaluate the returns distribution: '))
    print('generating samples for model evaluation')
    real, generated, it = evaluate_model.samples(dataset, 
                            sample_generator = df_generator,
                            n_samples=eval_sample)
    
    print(f'syntetic data generated within {it} iterations')

    print('saving samples distributions')
    evaluate_model.mean_average(generated.ravel(),0.05)
    evaluate_model.save_samples_dist(real, generated, destination_f)
    




    



    

