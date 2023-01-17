# sourced from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/Variational_autoencoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

latent_dim = 20

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

class Decoder(nn.Sequential):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(latent_dim, 400)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(400, 784)

    def forward(self, z):
        h1 = self.fc1(z)
        z1 = self.relu(h1)
        h2 = self.fc2(z1)
        h2 = h2.view(h2.size(0), 1, 28, 28)
        return h2

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.decoder(self.reparametrize(mu, logvar))
        return z, mu, logvar

def vae():
    return VAE()