# Imports
import itertools
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from lion_pytorch import Lion
import json

from vae import VAE
from finetuner import Finetuner

# Parameters to set:



n_epochs = 3000

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# load data and marginals


# Load model
group_sizes_delaware = [
    2,
    5,
    5,
    11,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
]

torch.manual_seed(2)
torch.cuda.manual_seed(2)

# Delaware
os.makedirs('/workspace/finetuned_models', exist_ok=True)
model_names = ['model_l25_h750_b2_g2_y21_final.pth']

for model_name in model_names:

    if 'north_carolina' in model_name:
        latent_dim = 25
        hidden_dim = 750
        input_dim = 607
        pums_data = pd.read_csv("/workspace/data/north_carolina_21.csv")

        with open("/workspace/data/ACS_tract_tables/NorthCarolina201/marginals.json") as f:
            marginals = json.load(f)

        prefixes = [v.split(':')[0] for v in pums_data.columns]
        group_sizes = [len(list(group)) for _, group in itertools.groupby(prefixes)]
    else:
        # Find latent_dim and hidden_dim from model_name.
        model_parts = model_name.split('_')
        latent_dim = int(model_parts[1][1:])
        hidden_dim = int(model_parts[2][1:])
        input_dim = 433
        pums_data = pd.read_csv("/workspace/data/one_hot_pNaNs_agep_21.csv")

        with open("/workspace/data/ACS_tract_tables/Delaware50101/marginals.json") as f:
            marginals = json.load(f)


        prefixes = [v.split(':')[0] for v in pums_data.columns]
        group_sizes = [len(list(group)) for _, group in itertools.groupby(prefixes)]

    model = VAE(input_dim, hidden_dim, 6, latent_dim, group_sizes)

    params = torch.load(f"/workspace/models/{model_name}", map_location=torch.device("cpu"))
    model.load_state_dict(params)

    # Generate synthetic codes - make trainable and put in optimizer
    
    
    finetuned_model_name = 'finetuned_' + model_name
    # Set weights
    # weights_list = [torch.tensor([1,25,2]), torch.tensor([1,1,1]), torch.tensor([1,1,100]), torch.tensor([1,1,10])]
    weights = torch.tensor([1,1,10])
    lr_0 = 1e-1
    lr_1 = 1e-2

    

    for n_samples in [50,100,200,500,1000]:
        trainable_latent_codes = torch.randn(n_samples, latent_dim).to(device)
        trainable_latent_codes.requires_grad = True

        optimizer = optim.AdamW([trainable_latent_codes], lr=lr_0)

        # Initialise finetuner and train
        finetuner = Finetuner(pums_data, marginals, model, optimizer, lr_0, lr_1, device)

        print(f'\nTraining on {model_name} with num samples {n_samples}')
        finetuner.train(trainable_latent_codes, n_epochs, weights, f"/workspace/finetuned_models/{finetuned_model_name}", n_samples = n_samples)
