import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)


    def kl_loss(self, mu, logvar):
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
    
    def focal_loss(self, x, x_hat, gamma=2, alpha=0.25):
        return -torch.mean(alpha * x_hat * (1-x)**gamma * torch.log(x) + (1-alpha) * (1-x_hat) * x**gamma * torch.log(1-x))
    # pre_training uses the focal loss and KL divergence loss
    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            for i, x in enumerate(train_loader):
                x = x.to(self.device)
                x_hat, mu, logvar = self.model(x)
                kl_loss = self.kl_loss(mu, logvar)
                focal_loss = self.focal_loss(x, x_hat)
                
                print(f'KL Loss: {kl_loss.item()}, Focal Loss: {focal_loss.item()}')

                loss = kl_loss + focal_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 100 == 0:
                    print(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()}')
            break