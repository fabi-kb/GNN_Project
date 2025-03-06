# module for the VAE model

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, latent_dim, group_sizes):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dim, hidden_layers, latent_dim)
        self.decoder = Decoder(
            input_dim, hidden_dim, hidden_layers, latent_dim, group_sizes
        )

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        return self.decoder(z), mu, logvar

    def pretrain_sample(self, num_samples):
        z = torch.randn((num_samples, self.latent_dim))
        return self.decoder(z)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, latent_dim):
        super(Encoder, self).__init__()
        self.layers = self._build_layers(
            input_dim, hidden_dim, hidden_layers, latent_dim
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.mu_bn = nn.BatchNorm1d(latent_dim, eps=1e-5)
        self.logvar_bn = nn.BatchNorm1d(latent_dim, eps=1e-5)

    def _build_layers(self, input_dim, hidden_dim, hidden_layers, latent_dim):
        layers = []
        layers.append(MLPBlock(input_dim, hidden_dim))
        for i in range(hidden_layers):
            layers.append(ResidualBlock(hidden_dim, hidden_dim))

        return nn.Sequential(*layers)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.layers(x)

        mu = self.mu_bn(self.mu(h))
        logvar = self.logvar_bn(self.logvar(h))
        return self.reparametrize(mu, logvar), mu, logvar


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, latent_dim, group_sizes):
        super(Decoder, self).__init__()
        self.layers = self._build_layers(
            input_dim, hidden_dim, hidden_layers, latent_dim, group_sizes
        )

    def _build_layers(
        self, input_dim, hidden_dim, hidden_layers, latent_dim, group_sizes
    ):
        layers = []
        layers.append(MLPBlock(latent_dim, hidden_dim))
        for i in range(hidden_layers):
            layers.append(ResidualBlock(hidden_dim, hidden_dim))

        layers.append(nn.Linear(hidden_dim, input_dim))
        layers.append(nn.BatchNorm1d(input_dim, eps=1e-5))
        layers.append(nn.ReLU())

        # apply softmax to get probabilities for each onehot encoded feature
        layers.append(GroupSoftmax(group_sizes))
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


class GroupSoftmax(nn.Module):
    def __init__(self, group_sizes):
        super(GroupSoftmax, self).__init__()
        self.group_sizes = group_sizes

    def forward(self, x):
        outputs = []
        start = 0
        for size in self.group_sizes:
            tmp = x[:, start : start + size]
            tmp = F.softmax(tmp, dim=1)
            outputs.append(tmp)
            start += size
        return torch.cat(outputs, dim=1)
