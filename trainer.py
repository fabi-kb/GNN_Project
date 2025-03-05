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

    def focal_loss(self, x, x_hat, gamma=2, alpha=0.125):
        eps = 1e-7
        x_hat = torch.clamp(x_hat, eps, 1 - eps)
        return -torch.mean(
            alpha * x * (1 - x_hat) ** gamma * torch.log(x_hat)
            + (1 - alpha) * (1 - x) * x_hat**gamma * torch.log(1 - x_hat)
        )

    # pre_training uses the focal loss and KL divergence loss
    def train(self, train_loader, epochs, disk_path=None, gamma=2, alpha=0.125):
        self.model.train()
        losses = []
        for epoch in range(epochs):

            for i, x in enumerate(train_loader):
                x = x.to(self.device)
                x_hat, mu, logvar = self.model(x)
                kl_loss = self.kl_loss(mu, logvar)
                focal_loss = self.focal_loss(x, x_hat, gamma, alpha)

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

            # decay learning
            if epoch >= self.start_decay and epoch <= self.stop_decay:
                self.new_lr = self.init_lr * self.decay_rate ** (
                    epoch - self.start_decay
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.new_lr

            print(f"Epoch {epoch}, Loss: {loss.item()}, LR: {self.new_lr}")


class Finetuner:
    def __init__(self, column_names, marginals, model, optimizer, device):
        self.column_names = column_names
        self.marginals = marginals

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        self.best_model = None

        self.n_marginal_vars = len(marginals)
        self.matching_indices = self._get_matching_indices(marginals, column_names)

    def kl_loss(self, p):
        n = len(p)  # uniform distribution would be 1/n for each index in p

        p = torch.clip(p, 1e-10, 1)  # clip to avoid log(0)
        return torch.sum(p * torch.log(n * p))

    def _get_matching_indices(self, marginals, column_names):
        """
        Get the indices of the columns in the data tensor that match the marginals
        """
        matching_indices = {v: [] for v in marginals.keys()}

        # Handle simple household variables
        common_vars = [
            var_name for var_name in marginals.keys() if var_name in column_names
        ]

        simple_vars = {
            var_name: column_names.index(var_name) for var_name in common_vars
        }

        matching_indices.update(simple_vars)

        # Handle household income variables that need extra binning
        extra_binning_vars = {
            "HINCP:under 10k": [
                idx
                for idx, col in enumerate(column_names)
                if col in ["HINCP:under 5k", "HINCP:5k-10k"]
            ],
            "HINCP:15k-25k": [
                idx
                for idx, col in enumerate(column_names)
                if col in ["HINCP:15k-20k", "HINCP:20k-25k"]
            ],
        }

        matching_indices.update(extra_binning_vars)

        # Handle personal variables
        # Find column indices to permit aggregating personal variables
        personal_one_hot_vars = [
            var.replace("_1", "") for var in column_names if "_1:" in var
        ]
        personal_var_indices = {}

        # now store which columns correspond to each personal variable
        for var in personal_one_hot_vars:
            var_parts = var.split(":")
            pattern = re.compile(r"{}_\d+:{}".format(var_parts[0], var_parts[1]))

            matching_indices = [
                idx for idx, col in enumerate(column_names) if pattern.match(col)
            ]
            personal_var_indices[var] = matching_indices

        matching_indices.update(personal_var_indices)

        return matching_indices

    def DBCE(self, predictions, labels):
        """
        Compute Decoupled binary cross-entropy
        """

        N = labels.shape[0]
        Nt = predictions.shape[0]

        # Initialise helper variables
        softIndex = torch.zeros(N)
        softminloss = 0
        bce = torch.zeros([Nt, N])

        # TODO: Replace with vectorised methods once I can show it doesn't change the result
        for i, x_hat in enumerate(predictions):
            for j, x in enumerate(labels):
                bce[i, j] = F.binary_cross_entropy(x_hat, x)

            softIndex_case = F.softmin(bce[i])
            softminloss += sum(bce[i] * softIndex_case[i])
            softIndex += softIndex_case

        DBCE = softminloss / Nt
        DBCEKL = self.kl_loss(softIndex, torch.ones(N) / N)

        return DBCE, DBCEKL

    def marginal_loss(self, predictions):
        sum_of_squares = 0

        for var, indices in self.matching_indices.items():
            predicted_marginal = predictions[:, indices].sum()
            sum_of_squares += (predicted_marginal - self.marginals[var]) ** 2

        RMSE = torch.sqrt(sum_of_squares / self.n_marginal_vars)

        return RMSE

    def train(
        self, trainable_latent_codes, epochs, disk_path=None, gamma=2, alpha=0.125
    ):
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_latent_codes.to(self.device)

        losses = []
        for epoch in range(epochs):
            # Predict from latent codes
            predictions = self.model.decoder(trainable_latent_codes)

            # Obtain loss (unweighted sum?)
            DBCE, DBCEKL = self.DBCE(predictions, self.data)
            marginal_loss = self.marginal_loss(predictions)
            loss = DBCE + DBCEKL + marginal_loss

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # save best model so far based on loss
            losses.append(loss.item())
            if not self.best_model or loss.item() < min(losses):
                self.best_model = self.model.state_dict()

            # save the best model to disk every 200 epochs
            if epoch % 200 == 0:
                torch.save(self.best_model, f"{disk_path}")


class Finetuner:
    def __init__(self, column_names, marginals, model, optimizer, device):
        self.column_names = column_names
        self.marginals = marginals

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        self.best_model = None

        self.n_marginal_vars = len(marginals)
        self.matching_indices = self._get_matching_indices(marginals, column_names)

    def kl_loss(self, p):
        n = len(p)  # uniform distribution would be 1/n for each index in p

        p = torch.clip(p, 1e-10, 1)  # clip to avoid log(0)
        return torch.sum(p * torch.log(n * p))

    def _get_matching_indices(self, marginals, column_names):
        """
        Get the indices of the columns in the data tensor that match the marginals
        """
        matching_indices = {v: [] for v in marginals.keys()}

        # Handle simple household variables
        common_vars = [
            var_name for var_name in marginals.keys() if var_name in column_names
        ]

        simple_vars = {
            var_name: column_names.index(var_name) for var_name in common_vars
        }

        matching_indices.update(simple_vars)

        # Handle household income variables that need extra binning
        extra_binning_vars = {
            "HINCP:under 10k": [
                idx
                for idx, col in enumerate(column_names)
                if col in ["HINCP:under 5k", "HINCP:5k-10k"]
            ],
            "HINCP:15k-25k": [
                idx
                for idx, col in enumerate(column_names)
                if col in ["HINCP:15k-20k", "HINCP:20k-25k"]
            ],
        }

        matching_indices.update(extra_binning_vars)

        # Handle personal variables
        # Find column indices to permit aggregating personal variables
        personal_one_hot_vars = [
            var.replace("_1", "") for var in column_names if "_1:" in var
        ]
        personal_var_indices = {}

        # now store which columns correspond to each personal variable
        for var in personal_one_hot_vars:
            var_parts = var.split(":")
            pattern = re.compile(r"{}_\d+:{}".format(var_parts[0], var_parts[1]))

            matching_indices = [
                idx for idx, col in enumerate(column_names) if pattern.match(col)
            ]
            personal_var_indices[var] = matching_indices

        matching_indices.update(personal_var_indices)

        return matching_indices

    def DBCE(self, predictions, labels):
        """
        Compute Decoupled binary cross-entropy
        """

        N = labels.shape[0]
        Nt = predictions.shape[0]

        # Initialise helper variables
        softIndex = torch.zeros(N)
        softminloss = 0
        bce = torch.zeros([Nt, N])

        # TODO: Replace with vectorised methods once I can show it doesn't change the result
        for i, x_hat in enumerate(predictions):
            for j, x in enumerate(labels):
                bce[i, j] = F.binary_cross_entropy(x_hat, x)

            softIndex_case = F.softmin(bce[i])
            softminloss += sum(bce[i] * softIndex_case[i])
            softIndex += softIndex_case

        DBCE = softminloss / Nt
        DBCEKL = self.kl_loss(softIndex, torch.ones(N) / N)

        return DBCE, DBCEKL

    def marginal_loss(self, predictions):
        sum_of_squares = 0

        for var, indices in self.matching_indices.items():
            predicted_marginal = predictions[:, indices].sum()
            sum_of_squares += (predicted_marginal - self.marginals[var]) ** 2

        RMSE = torch.sqrt(sum_of_squares / self.n_marginal_vars)

        return RMSE

    def train(
        self, trainable_latent_codes, epochs, disk_path=None, gamma=2, alpha=0.125
    ):
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_latent_codes.to(self.device)

        losses = []
        for epoch in range(epochs):
            # Predict from latent codes
            predictions = self.model.decoder(trainable_latent_codes)

            # Obtain loss (unweighted sum?)
            DBCE, DBCEKL = self.DBCE(predictions, self.data)
            marginal_loss = self.marginal_loss(predictions)
            loss = DBCE + DBCEKL + marginal_loss

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # save best model so far based on loss
            losses.append(loss.item())
            if not self.best_model or loss.item() < min(losses):
                self.best_model = self.model.state_dict()

            # save the best model to disk every 200 epochs
            if epoch % 200 == 0:
                torch.save(self.best_model, f"{disk_path}")


class Finetuner:
    def __init__(self, pums_data, marginals, model, optimizer, device):
        self.data = torch.tensor(pums_data.values).float().to(device)
        self.column_names = list(pums_data.columns)
        self.marginals = marginals

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        self.best_model = None

        self.n_marginal_vars = len(marginals)
        self.matching_indices = self._get_matching_indices(marginals, self.column_names)

    def kl_loss(self, p):
        n = len(p)  # uniform distribution would be 1/n for each index in p

        p = torch.clip(p, 1e-10, 1)  # clip to avoid log(0)
        return torch.sum(p * torch.log(n * p))

    def _get_matching_indices(self, marginals, column_names):
        """
        Get the indices of the columns in the data tensor that match the marginals
        """
        matching_indices = {v: [] for v in marginals.keys()}

        # Handle simple household variables
        common_vars = [
            var_name for var_name in marginals.keys() if var_name in column_names
        ]

        simple_vars = {
            var_name: column_names.index(var_name) for var_name in common_vars
        }

        matching_indices.update(simple_vars)

        # Handle household income variables that need extra binning
        extra_binning_vars = {
            "HINCP:under 10k": [
                idx
                for idx, col in enumerate(column_names)
                if col in ["HINCP:under 5k", "HINCP:5k-10k"]
            ],
            "HINCP:15k-25k": [
                idx
                for idx, col in enumerate(column_names)
                if col in ["HINCP:15k-20k", "HINCP:20k-25k"]
            ],
        }

        matching_indices.update(extra_binning_vars)

        # Handle personal variables
        # Find column indices to permit aggregating personal variables
        personal_one_hot_vars = [
            var.replace("_1", "") for var in column_names if "_1:" in var
        ]
        personal_var_indices = {}

        # now store which columns correspond to each personal variable
        for var in personal_one_hot_vars:
            var_parts = var.split(":")
            pattern = re.compile(r"{}_\d+:{}".format(var_parts[0], var_parts[1]))

            personal_matching_indices = [
                idx for idx, col in enumerate(column_names) if pattern.match(col)
            ]
            personal_var_indices[var] = personal_matching_indices

        matching_indices.update(personal_var_indices)

        return matching_indices

    # def DBCE(self, predictions, labels):
    #     """
    #     Compute Decoupled binary cross-entropy
    #     """

    #     N = labels.shape[0]
    #     Nt = predictions.shape[0]

    #     # Initialise helper variables
    #     softIndex = torch.zeros(N)
    #     softminloss = 0
    #     bce = torch.zeros([Nt, N])

    #     # TODO: Replace with vectorised methods once I can show it doesn't change the result
    #     for i, x_hat in enumerate(predictions):
    #         for j, x in enumerate(labels):
    #             bce[i, j] = F.binary_cross_entropy(x_hat, x)

    #         softIndex_case = F.softmin(bce[i])
    #         softminloss += sum(bce[i] * softIndex_case[i])
    #         softIndex += softIndex_case

    #     DBCE = softminloss / Nt
    #     DBCEKL = self.kl_loss(softIndex, torch.ones(N) / N)

    #     return DBCE, DBCEKL

    # def marginal_loss(self, predictions):
    #     sum_of_squares = 0

    #     for var, indices in self.matching_indices.items():
    #         predicted_marginal = predictions[:, indices].sum()
    #         sum_of_squares += (predicted_marginal - self.marginals[var]) ** 2

    #     RMSE = torch.sqrt(sum_of_squares / self.n_marginal_vars)

    #     return RMSE

    def DBCE(self, predictions, labels):
        """
        Compute Decoupled binary cross-entropy (vectorized version).

        Assumes:
        - predictions: tensor of shape (Nt, D)
        - labels: tensor of shape (N, D)
        """
        Nt = predictions.shape[0]  # number of predictions
        N = labels.shape[0]  # number of labels

        # predictions_exp: (Nt, 1, D) and labels_exp: (1, N, D)
        predictions_exp = predictions.unsqueeze(1).expand(Nt, N, -1)
        labels_exp = labels.unsqueeze(0).expand(Nt, N, -1)

        # Compute binary cross-entropy for all pairs, without reduction:
        bce_elem = F.binary_cross_entropy(
            predictions_exp, labels_exp, reduction="none"
        )  # Nt x N x D

        # Average over the feature dimension D to get a scalar per pair:
        bce = bce_elem.mean(dim=2)  # Nt x N

        # Compute the softmin of the BCE values for each prediction (each row)
        softIndex_all = F.softmin(bce, dim=1)  # Nt x N

        # Compute the weighted sum of BCE values for each prediction, then average over predictions.
        DBCE = (bce * softIndex_all).sum(dim=1).mean()

        # Sum the soft indices across all predictions to get a single vector for all labels.
        softIndex = softIndex_all.sum(dim=0)  # shape: (N,)

        # Compute KL divergence between softIndex and the uniform distribution.
        DBCEKL = self.kl_loss(softIndex)

        return DBCE, DBCEKL

    def train(
        self, trainable_latent_codes, epochs, disk_path=None, gamma=2, alpha=0.125
    ):
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_latent_codes.to(self.device)

        losses = []
        for epoch in range(epochs):
            # Predict from latent codes
            predictions = self.model.decoder(trainable_latent_codes)

            # Obtain loss (unweighted sum?)
            DBCE, DBCEKL = self.DBCE(predictions, self.data)
            marginal_loss = self.marginal_loss(predictions)
            loss = DBCE + DBCEKL + marginal_loss

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # save best model so far based on loss
            losses.append(loss.item())
            if not self.best_model or loss.item() < min(losses):
                self.best_model = self.model.state_dict()

            # save the best model to disk every 200 epochs
            if epoch % 200 == 0:
                torch.save(self.best_model, f"{disk_path}")

            print(f"Epoch {epoch}, Loss: {loss.item()}, LR: {self.new_lr}")
