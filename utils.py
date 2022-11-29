import os
import shutil
import torch

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