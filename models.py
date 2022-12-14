from torch import nn
from torch.nn.utils import spectral_norm

class AddDimension(nn.Module):
    def forward(self, x):
        #turn 1*n array into n*1 attay
        #is like the flatten layer in tensorflow
        return x.unsqueeze(1)

class SqueezeDimension(nn.Module):
    def forward(self, x):
        #turn n*1 array into a 1*n array
        return x.squeeze(1)

#build the generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.lin1  = nn.Linear(50, 100) #apply linear transformation A^Ta +b
        self.relu1 = nn.LeakyReLU(0.2, inplace=True) #apply poinwise LeakyReLU function
        self.dim1  = AddDimension() # flatten

        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.usam1 = nn.Upsample(200)
        self.drop1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv1d(32, 32, 3, padding=1)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.usam2 = nn.Upsample(400)
        self.drop2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv1d(32, 32, 3, padding=1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.usam3 = nn.Upsample(800)
        self.drop3 = nn.Dropout(0.1)

        self.conv4 = nn.Conv1d(32, 1, 3, padding=1)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.sq   = SqueezeDimension()
        self.lin2 = nn.Linear(800, 100)

        self.initialize_weights()

    def forward(self, input):

        out = self.lin1(input)
        out = self.relu1(out)
        out = self.dim1(out)
        #apply spectral normalization
        out = self.conv1(out)
        out = self.usam1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.usam2(out)
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.usam3(out)
        out = self.drop3(out)

        out = self.conv4(out)
        out = self.relu4(out)

        out = self.sq(out)
        out = self.lin2(out)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

#build the discriminator
class Critic(nn.Module):
    def __init__(self):

        super(Critic, self).__init__()
        
        self.adim  = AddDimension()
        self.conv1 = spectral_norm(nn.Conv1d(1, 32, 3, padding=1), n_power_iterations=10)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.mpol1 = nn.MaxPool1d(2)

        self.conv2 = spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.mpol2 = nn.MaxPool1d(2)

        self.conv3 = spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.flttn = nn.Flatten()

        self.lin1  = nn.Linear(800, 50)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.lin2  = nn.Linear(50, 15)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        
        # the output of the last layer is a single number as we can see by the last argument of the following function
        self.lin3  = nn.Linear(15, 1)

    def forward(self, input):

        out = self.adim(input)
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.mpol1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.mpol2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.flttn(out)

        out = self.lin1(out)
        out = self.relu4(out)

        out = self.lin2(out)
        out = self.relu5(out)
        
        # the output of the last layer is a single number as we can see by the last argument of the following function
        out = self.lin3(out)

        return out