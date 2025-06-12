import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda:0"
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from dataset.dataset import Dataset
from models.clim_regressor import ClimRegressor

# General settings
run_name = "clim_regression_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
n_epoch = 50
batch_size = 128
n_channels = 4
n_clim = 21
lrate = 5e-5

# Prepare output directory
output_dir = os.path.join("..", "out", run_name)
try:
    os.makedirs(output_dir)
except FileExistsError:
    print(f"{output_dir} already exists. Its contents may be overwritten.")

# Data stuff
data_root = os.path.join("..", "data")
regions = gpd.read_file(os.path.join(data_root, "lucas_regions.gpkg"))
regions_train = regions[regions.split == "train"]
regions_val = regions[regions.split == "val"]
train_ds = []
for country, region in zip(regions_train.country, regions_train.region):
    train_ds.append(Dataset(data_root, country, region))
train_ds = ConcatDataset(train_ds)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
val_ds = []
for country, region in zip(regions_val.country, regions_val.region):
    val_ds.append(Dataset(data_root, country, region))
val_ds = ConcatDataset(val_ds)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)

# Model stuff
regressor = ClimRegressor(n_channels=n_channels, n_clim=n_clim).to(device)
optim = torch.optim.AdamW(regressor.parameters(), lr=lrate, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "max")

# Training
train_loss_tracker, val_loss_tracker = [], []
min_val_loss = np.inf

for ep in range(n_epoch):

    cum_loss = 0
    batch_count = 0

    # Training loop
    regressor.train()
    pbar = tqdm(train_dl, desc=f'Training epoch {ep}')
    for im, clim, _, _, _, _ in pbar:
        optim.zero_grad()
        im, clim = im.to(device), clim.to(device)
        bc_hat = regressor(im)
        loss = F.mse_loss(bc_hat, clim)
        loss.backward()
        optim.step()
        cum_loss += loss.item()
        batch_count += 1
    train_loss_tracker.append(cum_loss / batch_count)
    
    # Validation loop
    regressor.eval()
    pbar = tqdm(val_dl, desc=f"Validation epoch {ep}")
    cum_loss = 0
    batch_count = 0
    for im, clim, _, _, _, _ in pbar:
        im, clim = im.to(device), clim.to(device)
        with torch.inference_mode():
            bc_hat = regressor(im)
            loss = F.mse_loss(bc_hat, clim)
        cum_loss += loss.item()
        batch_count += 1
    val_loss_tracker.append(cum_loss / batch_count)
    sched.step(cum_loss / batch_count)
    
    # Saving intermediate results
    plt.figure(figsize=(10,10))
    plt.plot(np.arange(0, ep+1), train_loss_tracker, label="Training loss")
    plt.plot(np.arange(0, ep+1), val_loss_tracker, label="Validation loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    if (cum_loss / batch_count) < min_val_loss:
        torch.save(regressor.state_dict(), os.path.join(output_dir, f"model.pt"))
        min_val_loss = cum_loss / batch_count