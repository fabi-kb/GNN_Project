# Imports
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
model_name = "model_l100_h500_b5_g2.pth"

# Find latent_dim and hidden_dim from model_name.
model_parts = model_name.split('_')
latent_dim = int(model_parts[1][1:])
hidden_dim = int(model_parts[2][1:])

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

model = VAE(433, hidden_dim, 6, latent_dim, group_sizes)

params = torch.load(f"/workspace/models/{model_name}", map_location=torch.device("cpu"))
model.load_state_dict(params)

# Generate synthetic codes - make trainable and put in optimizer
trainable_latent_codes = torch.randn(n_synthetic_samples, latent_dim).to(device)
trainable_latent_codes.requires_grad = True

optimizer = optim.AdamW([trainable_latent_codes], lr=1e-3)

# Initialise finetuner and train
finetuner = Finetuner(pums_data, marginals, model, optimizer, device)

finetuner.train(trainable_latent_codes, 100, "workspace/finetuned_models")
