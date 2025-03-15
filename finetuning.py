# Imports
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
n_synthetic_samples = 500
data_name = "one_hot_pNaNs_agep.csv"
model_name = "bms_l75_h500_b1_g2.pth"
all_models_names = os.listdir('/workspace/models')
model_names = [name for name in all_models_names if name.startswith('bms')]

# # Don't repeat already trained models
# os.makedirs('/workspace/finetuned_models', exist_ok=True)
# model_names = [name for name in all_models_names if not 'finetuned_full' + name in os.listdir('/workspace/finetuned_models')]

n_epochs = 2000

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# load data and marginals
pums_data = pd.read_csv(f"/workspace/data/{data_name}")

with open("marginals.json") as f:
    marginals = json.load(f)

# Load model
group_sizes = [
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

torch.manual_seed(1)
torch.cuda.manual_seed(1)

os.makedirs('/workspace/finetuned_models', exist_ok=True)
# TESTING!
for model_name in model_names:
    model_name = 'bms_l100_h500_b2_g2.pth'
    print(f'\nTraining on {model_name}')
    # Find latent_dim and hidden_dim from model_name.
    model_parts = model_name.split('_')
    latent_dim = int(model_parts[1][1:])
    hidden_dim = int(model_parts[2][1:])

    model = VAE(433, hidden_dim, 6, latent_dim, group_sizes)

    params = torch.load(f"/workspace/models/{model_name}", map_location=torch.device("cpu"))
    model.load_state_dict(params)

    # Generate synthetic codes - make trainable and put in optimizer
    trainable_latent_codes = torch.randn(n_synthetic_samples, latent_dim).to(device)
    trainable_latent_codes.requires_grad = True

    lr_0 = 5e-1
    lr_1 = 1e-2
    optimizer = optim.AdamW([trainable_latent_codes], lr=lr_0)

    # Initialise finetuner and train
    finetuner = Finetuner(pums_data, marginals, model, optimizer, lr_0, lr_1, device)

    
    finetuned_model_name = 'finetuned_' + model_name
    finetuner.train(trainable_latent_codes, n_epochs, f"/workspace/finetuned_models/{finetuned_model_name}")
