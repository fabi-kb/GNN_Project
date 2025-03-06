import re
import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        self.best_model = None

        # LR decay schedule
        self.start_decay = 1000
        self.stop_decay = 4000
        self.init_lr = 1e-3
        self.new_lr = self.init_lr
        self.final_lr = 1e-4
        self.decay_rate = (self.final_lr / self.init_lr) ** (
            1.0 / (self.stop_decay - self.start_decay)
        )

    def kl_loss(self, mu, logvar):
        return 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1)

    def focal_loss(self, x, x_hat, gamma, alpha):
        tmp = alpha * x * (1 - x_hat).pow(gamma) * torch.log(x_hat + 1e-7) + (
            1 - alpha
        ) * (1 - x) * x_hat.pow(gamma) * torch.log(1 - x_hat + 1e-7)

        tmp_sum = torch.sum(tmp, dim=1)
        # print(tmp_sum.shape)
        return -torch.mean(tmp_sum)

    # pre_training uses the focal loss and KL divergence loss
    def train(
        self, train_loader, epochs, disk_path=None, gamma=2, alpha=0.125, beta=5.0
    ):
        self.model.train()
        losses = []
        init_beta = beta
        for epoch in range(epochs):
            if epoch < self.start_decay:
                beta = init_beta + (2.0 - init_beta) * (epoch / self.start_decay)
            else:
                beta = 2.0
            for i, x in enumerate(train_loader):
                x = x.to(self.device)
                x_hat, mu, logvar = self.model(x)
                kl_loss = self.kl_loss(mu, logvar)
                focal_loss = self.focal_loss(x, x_hat, gamma, alpha)

                loss = beta * kl_loss + focal_loss
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

            # decay learning
            if epoch >= self.start_decay and epoch <= self.stop_decay:
                self.new_lr = self.init_lr * self.decay_rate ** (
                    epoch - self.start_decay
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.new_lr

            print(
                f"Epoch {epoch}, Loss: {loss.item()}, LR: {self.new_lr}, Beta: {beta}, KL Loss: {beta * kl_loss}, Focal Loss: {focal_loss}"
            )
