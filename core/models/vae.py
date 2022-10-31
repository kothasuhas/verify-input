import torch
import torch.nn as nn
import torch.nn.functional as F

latent_dims = 2
num_epochs = 100
batch_size = 1
capacity = 64
learning_rate = 1e-3
variational_beta = 5
use_gpu = True

ndim = 256

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.fc1 = nn.Linear(in_features=ndim, out_features=ndim//2)
        self.fc2 = nn.Linear(in_features=ndim//2, out_features=latent_dims*2)
        self.fc_mu = nn.Linear(in_features=latent_dims*2, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=latent_dims*2, out_features=latent_dims)
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=latent_dims*2)
        self.fc2 = nn.Linear(in_features=latent_dims*2, out_features=ndim//2)
        self.fc1 = nn.Linear(in_features=ndim//2, out_features=ndim)
            
    def forward(self, x):
        x = self.fc(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

def vae():
    return VariationalAutoencoder()