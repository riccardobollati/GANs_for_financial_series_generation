import os

import torch
from tqdm import tqdm
from torch.autograd import grad as torch_grad
import matplotlib.pyplot as plt


class Trainer:
    
    NOISE_LENGTH = 50

    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                gp_weight, g_norm_pen, critic_iterations, print_every, checkpoint_frequency, writer, ARCHIVE_DIR):
                
        self.g = generator
        self.g_opt = gen_optimizer
        self.c = critic
        self.c_opt = critic_optimizer
        self.losses = {'g': [], 'c': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        #if True it will compute on GPU (but unavailable for my pc)
        self.gp_weight = gp_weight
        self.g_norm_pen = g_norm_pen
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.checkpoint_frequency = checkpoint_frequency
        self.writer = writer
        self.ARCHIVE_DIR = ARCHIVE_DIR

    def _critic_train_iteration(self, real_data):
        '''
        train the discriminator for one iteration
        :param real_data: batches of real samples 
        '''

        batch_size = real_data.size()[0]

        # generate the fake samples-----------------------------------------
        # shape of the generator input
        noise_shape = (batch_size, self.NOISE_LENGTH)
        # generate fake samples
        generated_data = self.sample_generator(noise_shape)

        #turn eral data in to tensors
        #real_data = torch.tensor(real_data)

        # Pass data through the Critic
        # predict real data labels
        c_real = self.c(real_data)
        # predict fake samples labels
        c_generated = self.c(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(real_data, generated_data)
        self.losses['GP'].append(gradient_penalty.data.item())

        # Create total loss and optimize

        # reset optimizer grads
        self.c_opt.zero_grad()
        # compute the loss
        d_loss = c_generated.mean() - c_real.mean() + gradient_penalty
        # compute the gradient of the given tensor
        d_loss.backward()

        # the following function perform a single optimization step
        self.c_opt.step()

        #save the loss in the dictionary
        self.losses['c'].append(d_loss.data.item())

    def _generator_train_iteration(self, data):
        '''
        train the generator for a single iteration
        :param data: real data, used just to take the number of batches
        '''

        # reset the optimizer grads 
        self.g_opt.zero_grad()

        batch_size = data.size()[0]
        # generate a random seed for the generator
        latent_shape = (batch_size, self.NOISE_LENGTH)

        # generate fake samples
        generated_data = self.sample_generator(latent_shape)

        # Calculate loss and optimize

        # test the generated data with the discriminator
        d_generated = self.c(generated_data)
        # compute average loss per batch
        g_loss = - d_generated.mean()
        # compute the gradients
        g_loss.backward()
        
        # perform a single opt step
        self.g_opt.step()
        #save the loss
        self.losses['g'].append(g_loss.data.item())

    def _gradient_penalty(self, real_data, generated_data):
        '''
        compute the gradient penalty using the improved wesserstain training
        '''

        batch_size = real_data.size()[0]

        # Calculate interpolation------------------------------
        # create a number of batch vectors of batch size where each entry is a random float between 0 and 1
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data)
        
        #I have removed .data after real_data (think is deprecated)
        interpolated = alpha * real_data + (1 - alpha) * generated_data.data
        #interpolated = torch.tensor(interpolated, requires_grad=True)
        interpolated = interpolated.detach().clone().requires_grad_()

        # Pass interpolated data through Critic
        prob_interpolated = self.c(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()), create_graph=True,
                               retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, series length),
        # here we flatten to take the norm per example for every batch
        gradients = gradients.view(batch_size, -1)
        #we append to the loss dictionary the mean of the norm of the grads
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data.item())

        # Derivatives of the gradient close to 0 can cause problems because of the
        # square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - self.g_norm_pen) ** 2).mean()

    def _train_epoch(self, data_loader, epoch):
        '''
        train both the discriminator and the generator for a single epoch
        :param data_loader: the dataset as data loader objct
        :param epoch: epoch number
        '''

        # for each sample in the dataset passed
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            
            # train the discriminator 
            self._critic_train_iteration(data.float())
            
            # Only update generator sometimes
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data)

                # refresh the tensorboard
                if i % self.print_every == 0:
                    global_step = i + epoch * len(data_loader.dataset)
                    self.writer.add_scalar('Losses/Critic', self.losses['c'][-1], global_step)
                    self.writer.add_scalar('Losses/Gradient Penalty', self.losses['GP'][-1], global_step)
                    self.writer.add_scalar('Gradient Norm', self.losses['gradient_norm'][-1], global_step)

                    if self.num_steps > self.critic_iterations:
                        self.writer.add_scalar('Losses/Generator', self.losses['g'][-1], global_step)

    def train(self, data_loader, epochs, plot_training_samples=True, checkpoint=None):
        '''
        this function train the model for n epochs
        :param data_loader: real data dataset as DataLoader object
        :param epochs: number of epochs
        :param plot_training_samples:
        :param checkpoint: checkpoint to load 
        '''

        if checkpoint:
            path = os.path.join('checkpoints', checkpoint)
            state_dicts = torch.load(path, map_location=torch.device('cpu'))
            self.g.load_state_dict(state_dicts['g_state_dict'])
            self.c.load_state_dict(state_dicts['d_state_dict'])
            self.g_opt.load_state_dict(state_dicts['g_opt_state_dict'])
            self.c_opt.load_state_dict(state_dicts['d_opt_state_dict'])

        # Define noise_shape
        noise_shape = (1, self.NOISE_LENGTH)

        if plot_training_samples:
            # Fix latents to see how series generation improves during training
            fixed_latents = torch.randn(noise_shape, requires_grad=True)
    

        for epoch in tqdm(range(epochs)):

            # Sample a different region of the latent distribution to check for mode collapse
            dynamic_latents = torch.randn(noise_shape, requires_grad=True)

            # train both the model for a single iteration
            self._train_epoch(data_loader, epoch + 1)

            # Save checkpoint
            if ((epoch+1) % self.checkpoint_frequency == 0) or (epoch == 0):
                torch.save({
                    'epoch': epoch,
                    'd_state_dict': self.c.state_dict(),
                    'g_state_dict': self.g.state_dict(),
                    'd_opt_state_dict': self.c_opt.state_dict(),
                    'g_opt_state_dict': self.g_opt.state_dict(),
                }, os.path.join(self.ARCHIVE_DIR, 'checkpoints/epoch_{}.pkl'.format(epoch)))

            if plot_training_samples and (epoch % self.print_every == 0):
                #set the module in evaluation mode
                self.g.eval()
                # Generate fake data using both fixed and dynamic latents
                fake_data_fixed_latents = self.g(fixed_latents).cpu().data
                # this latent changes for every epochs
                fake_data_dynamic_latents = self.g(dynamic_latents).cpu().data

                plt.figure()
                plt.plot(fake_data_fixed_latents.numpy()[0].T)
                plt.savefig(os.path.join(self.ARCHIVE_DIR, 'training_samples/fixed_latents/series_epoch_{}.png'.format(epoch)))
                plt.close()

                plt.figure()
                plt.plot(fake_data_dynamic_latents.numpy()[0].T)
                plt.savefig(os.path.join(self.ARCHIVE_DIR, 'training_samples/dynamic_latents/series_epoch_{}.png'.format(epoch)))
                plt.close()
                self.g.train()

    def sample_generator(self, latent_shape):
        '''
        generate a fake instance using the generator model
        :param latent_shape: shape of random data that the model takes as input
        '''
        #latent_samples = Variable(self.sample_latent(latent_shape))
        latent_samples = torch.randn(latent_shape, requires_grad=True)
        return self.g(latent_samples)

    def sample(self, num_samples):
        '''
        generate n fake samples uning the generator
        '''
        generated_samples = []
        for i in range(num_samples):
            generated_samples.append(self.sample_generator((1,self.NOISE_LENGTH)).data.numpy())

        return generated_samples
    