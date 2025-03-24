import json
import os
import re
import torch
import torch.nn.functional as F

def get_matching_indices(marginals, column_names):
        """
        Get the indices of the columns in the data tensor that match the marginals
        """
        # Household indices
        household_indices = {}

        # Handle simple household variables
        common_vars = [
            var_name for var_name in marginals.keys() if var_name in column_names
        ]

        simple_vars = {
            var_name: column_names.index(var_name) for var_name in common_vars
        }

        household_indices.update(simple_vars)

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

        household_indices.update(extra_binning_vars)

        # Handle personal variables
        # Find column indices to permit aggregating personal variables
        personal_one_hot_vars = [
            var.replace("_1", "") for var in column_names if "_1:" in var 
        ]
        personal_indices = {}
        nan_indices = {}

        # now store which columns correspond to each personal variable
        for var in personal_one_hot_vars:
            var_parts = var.split(":")
            pattern = re.compile(r"{}_\d+:{}".format(var_parts[0], var_parts[1]))

            matching_indices = [
                idx for idx, col in enumerate(column_names) if pattern.match(col)
            ]

            # Store in dictionary depending on if nan or not
            if 'nan' in var:
                nan_indices[var] = matching_indices
            else:
                personal_indices[var] = matching_indices

        return household_indices, personal_indices, nan_indices

class Finetuner:
    def __init__(self, pums_data, marginals, model, optimizer, lr_0, lr_1, device):
        self.data = torch.tensor(pums_data.values).float().to(device)
        self.column_names = list(pums_data.columns)
        self.marginals = marginals

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        self.best_model = None

        self.start_decay = 500
        self.stop_decay = 2500
        self.init_lr = lr_0
        self.new_lr = self.init_lr
        self.final_lr = lr_1
        self.decay_rate = (self.final_lr / self.init_lr) ** (
            1.0 / (self.stop_decay - self.start_decay)
        )

        self.n_marginal_vars = len(marginals)
        self.household_indices, self.personal_indices, self.nan_indices = get_matching_indices(marginals, self.column_names)
        self.non_conforming_column_indices = self._get_non_conf_col_indices(self.column_names)
        self.n_non_conf_columns = len(self.non_conforming_column_indices)

        self.n_households = torch.nan
        self.num_people = torch.nan

    def kl_loss(self, p):
        n = len(p)  # uniform distribution would be 1/n for each index in p

        p = torch.clip(p, 1e-8, 1)  # clip to avoid log(0)
        return torch.sum(p * torch.log(n * p))

    def _get_non_conf_col_indices(self, column_names):
        '''
        Get indices of columns that don't have a match in the marginals dict. For these columns, try to get values to 0 in finetuning
        '''

        schl_pattern = re.compile(r'SCHL_\d+:other')
        non_conf_indices = [idx for idx, col in enumerate(column_names) if 
                              'nan' in col
                              or schl_pattern.search(col)]

        return non_conf_indices
    
    def marginal_loss(self, predictions):
        sum_of_squares = 0
        # Handle household variables
        for var, indices in self.household_indices.items():
            predicted_marginal = predictions[:, indices].mean()
            sum_of_squares += (predicted_marginal - self.marginals[var]) ** 2

        # Handle personal variables
        n_people_per_household = len(list(self.personal_indices.values())[0])
        n_people_with_nans = n_people_per_household * self.n_households

        for var, indices in self.personal_indices.items():
            matching_nan_variable = var.split(":")[0] + ':' + 'nan' # Eg for SEX:male, matching nan variable is SEX:nan
            n_people = n_people_with_nans - predictions[:, self.nan_indices[matching_nan_variable]].sum()
            predicted_marginal = predictions[:, indices].sum()/n_people
            sum_of_squares += (predicted_marginal - self.marginals[var])**2

        RMSE = torch.sqrt(sum_of_squares / self.n_marginal_vars) 

        return RMSE

    def DBCE(self, predictions, labels):
        # predictions: (Nt, D) and labels: (N, D)
        Nt, D = predictions.shape
        eps = 1e-6  # for numerical stability

        # Clamp predictions to avoid log(0)
        predictions = torch.clamp(predictions, min=eps, max= 1 - eps)

        # Compute the logarithms
        log_p = torch.log(predictions)            # (Nt, D)
        log_1_minus_p = torch.log(1 - predictions)  # (Nt, D)

        # Compute the pairwise BCE using matrix multiplication:
        # First term: (Nt, D) @ (D, N) gives a (Nt, N) matrix.
        term1 = torch.matmul(log_p, labels.t())
        # Second term: (Nt, D) @ (D, N) for (1-labels)
        term2 = torch.matmul(log_1_minus_p, (1 - labels).t())
        
        # Combine the terms and average over D.
        # The minus sign is applied as in the BCE formula.
        bce = -(term1 + term2) / D  # shape: (Nt, N)
        
        # Compute the softmin along each prediction (row)
        softIndex_all = F.softmin(bce, dim=1)  # shape: (Nt, N)
        
        # Weighted sum of BCE values for each prediction, then average over predictions.
        DBCE = (bce * softIndex_all).sum(dim=1).mean()
        
        # Sum the soft indices across all predictions to get a single vector for all labels.
        softIndex = softIndex_all.sum(dim=0)/Nt  # shape: (N,)
        
        # Compute KL divergence between softIndex and the uniform distribution.
        DBCEKL = self.kl_loss(softIndex)
        
        return DBCE, DBCEKL


    def train(
        self, trainable_latent_codes, epochs, weights, disk_path=None, n_samples = None
    ):
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_latent_codes.to(self.device)

        # Store number of synthetic households
        self.n_households = trainable_latent_codes.shape[0]

        # Normalise weights
        alpha, beta, gamma = weights.numpy().astype(int).tolist()
        weights = weights.float()
        weights /= weights.sum()

        # Initialise loss trackers
        losses = []
        marginal_losses = []
        DBCE_losses = []
        DBCEKL_losses = []
        losses_dict = {'losses': losses,
                    'marginal_losses': marginal_losses,
                    'DBCE_losses': DBCE_losses,
                    'DBCEKL_losses': DBCEKL_losses}

        for epoch in range(epochs):
            # Predict from latent codes
            predictions = self.model.decoder(trainable_latent_codes)

            # Obtain loss

            DBCE, DBCEKL = self.DBCE(predictions, self.data)
            marginal_loss = self.marginal_loss(predictions)

            loss = weights[0]*DBCE + weights[1]*DBCEKL + weights[2]*marginal_loss

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Store losses
            losses.append(loss.item())
            marginal_losses.append(marginal_loss.item())
            DBCE_losses.append(DBCE.item())
            DBCEKL_losses.append(DBCEKL.item())

            # save best model so far based on loss
            if not self.best_model or loss.item() < min(losses):
                self.best_model = self.model.state_dict()


            # Decay learning rate - for now omit
            if epoch >= self.start_decay and epoch <= self.stop_decay:
                self.new_lr = self.init_lr * self.decay_rate ** (
                    epoch - self.start_decay
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.new_lr

            if epoch == 0:
                print(f'Initial WEIGHTED losses: DBCE: {alpha*DBCE.item():.5f}, DBCEKL: {beta*DBCEKL.item():.5f}, marginal: {gamma*marginal_loss.item():.5f}')

            if epoch % 10 == 0: 
                print(f"Epoch {epoch}, Loss: {loss.item():.5f}, DBCE: {DBCE.item():.5f}, DBCEKL: {DBCEKL.item():.5f}, marginal: {marginal_loss.item():.5f}")

        # save the best model at the end
        model_name = disk_path.split('/')[-1].removesuffix('.pth')
        
        torch.save(f'/workspace/finetuned_models/{model_name}_{alpha}_{beta}_{gamma}.pth', f"{disk_path}")

        # Store final losses
        os.makedirs('/workspace/finetuned_models/losses',exist_ok=True)
        if n_samples:
            with open(f'/workspace/finetuned_models/losses/{model_name}_{n_samples}.json', 'w') as f:
                json.dump(losses_dict, f, indent = 4)
        else:
            with open(f'/workspace/finetuned_models/losses/{model_name}_{alpha}_{beta}_{gamma}.json', 'w') as f:
                json.dump(losses_dict, f, indent = 4)
        # Store final predictions
        predictions_folder = '/workspace/finetuned_models/predictions/'
        os.makedirs(predictions_folder, exist_ok=True)
        final_preds = self.model.decoder(trainable_latent_codes)
        
        if n_samples:
            torch.save(final_preds, predictions_folder + model_name + f'_{n_samples}.pth')
        else:
            torch.save(final_preds, predictions_folder + model_name + f'_{alpha}_{beta}_{gamma}.pth')
            

