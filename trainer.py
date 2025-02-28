import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        self.best_model = None

    def kl_loss(self, mu, logvar):
        return 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1)

    def focal_loss(self, x, x_hat, gamma=2, alpha=0.17):
        eps = 1e-7
        x_hat = torch.clamp(x_hat, eps, 1 - eps)
        return -torch.mean(
            alpha * x * (1 - x_hat) ** gamma * torch.log(x_hat)
            + (1 - alpha) * (1 - x) * x_hat**gamma * torch.log(1 - x_hat)
        )

    # pre_training uses the focal loss and KL divergence loss
    def train(self, train_loader, epochs, disk_path=None):
        self.model.train()
        losses = []
        for epoch in range(epochs):

            for i, x in enumerate(train_loader):
                x = x.to(self.device)
                x_hat, mu, logvar = self.model(x)
                kl_loss = self.kl_loss(mu, logvar)
                focal_loss = self.focal_loss(x, x_hat)

                loss = kl_loss + focal_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # save best model so far based on loss
            losses.append(loss.item())
            if not self.best_model or loss.item() < min(losses):
                self.best_model = self.model.state_dict()

            print(f"Epoch {epoch}, Loss: {loss.item()}")
            # save the best model to disk every 200 epochs
            if epoch % 200 == 0:
                torch.save(self.best_model, f"{disk_path}")
