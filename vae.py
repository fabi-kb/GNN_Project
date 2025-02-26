# module for the VAE model 

import torch 
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, hidden_layers, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dim, hidden_layers, latent_dim)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        return self.decoder(z), mu, logvar

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, latent_dim):
        super(Encoder, self).__init__()
        self.layers = self._build_layers(input_dim, hidden_dim, hidden_layers, latent_dim)
    
    def _build_layers(self, input_dim, hidden_dim, hidden_layers, latent_dim):
        layers = []
        layers.append(MLPBlock(input_dim, hidden_dim))
        for i in range(hidden_layers):
            layers.append(ResidualBlock(hidden_dim, hidden_dim))
        # output mu and logvar
        layers.append(nn.Linear(hidden_dim, latent_dim*2))
        layers.append(nn.BatchNorm1d(latent_dim*2, eps=1e-5))
        return nn.Sequential(*layers)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.layers(x).chunk(2, dim=1)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, latent_dim):
        super(Decoder, self).__init__()
        self.layers = self._build_layers(input_dim, hidden_dim, hidden_layers, latent_dim)


    def _build_layers(self, input_dim, hidden_dim, hidden_layers, latent_dim):
        layers = []
        layers.append(MLPBlock(latent_dim, hidden_dim))
        for i in range(hidden_layers):
            layers.append(ResidualBlock(hidden_dim, hidden_dim))
        
        layers.append(nn.Linear(hidden_dim, input_dim))
        layers.append(nn.BatchNorm1d(input_dim, eps=1e-5))
        layers.append(nn.ReLU())

        # apply softmax to get probabilities for each onehot encoded feature -- idk if this is correct
        layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    

class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, eps=1e-5):
        super(MLPBlock, self).__init__()
        self.layers = self._build_layers(input_dim, hidden_dim, eps)

    def _build_layers(self, input_dim, hidden_dim, eps):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        # batch normalization
        layers.append(nn.BatchNorm1d(hidden_dim, eps=eps))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# consists of two MLP blocks and a skip connection
class ResidualBlock(nn.Module): 
    def __init__(self, input_dim, hidden_dim, eps=1e-5):
        super(ResidualBlock, self).__init__()
        self.mlp1 = MLPBlock(input_dim, hidden_dim, eps)
        self.mlp2 = MLPBlock(hidden_dim, input_dim, eps)
    
    def forward(self, x):
        return x + self.mlp2(self.mlp1(x))
    