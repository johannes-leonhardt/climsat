import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda:0"
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from models.diffusion import GaussianDiffusionModel
from dataset.dataset import Dataset

# General settings
n_epoch = 500
batch_size = 128
n_channels = 4
n_clim = 21
use_lc = True
n_lc = 9
n_T = 500
lrate = 5e-5
p_uncond = 0.1

# Prepare output directory
run_name = "diffusion_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_clim{'_lc' if use_lc else ''}"
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
val_ds = []
for country, region in zip(regions_val.country, regions_val.region):
    val_ds.append(Dataset(data_root, country, region))
val_ds = ConcatDataset(val_ds)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)

# Model stuff
if not use_lc:
    n_c = n_clim + 1
    normalization = "CBN"
else:
    n_c = [n_clim + 1, n_lc]
    normalization = "MCBN"
diffusion_model = GaussianDiffusionModel(n_channels=n_channels, betas=(1e-4, 0.02), n_T=n_T, device=device, n_c=n_c, normalization=normalization)
optim = torch.optim.Adam(diffusion_model.parameters(), lr=lrate)

# Training
train_loss_tracker, val_loss_tracker = [], []
min_val_loss = np.inf
for ep in range(n_epoch):

    # Training loop
    diffusion_model.train()
    optim.param_groups[0]['lr'] = lrate * (1 - ep/n_epoch)
    pbar = tqdm(train_dl, desc=f"Training epoch {ep}")
    cum_loss = 0
    batch_count = 0
    for im, clim, lc, _, _, _ in pbar:
        optim.zero_grad()
        im, clim, lc = im.to(device), clim.to(device), lc.to(device)
        if not use_lc:
            c = clim
        else:
            c = [clim, lc]
        loss = diffusion_model(im, c, p_uncond)
        loss.backward()
        cum_loss += loss.item()
        batch_count += 1
        optim.step()
    train_loss_tracker.append(cum_loss / batch_count)

    # Validation loop
    if ep % 10 == 0:
        diffusion_model.eval()
        pbar = tqdm(val_dl, desc=f"Validation epoch {ep}")
        cum_loss = 0
        batch_count = 0
        for im, clim, lc, _, _, _ in pbar:
            im, clim, lc = im.to(device), clim.to(device), lc.to(device)
            if not use_lc:
                c = clim
            else:
                c = [clim, lc]
            with torch.inference_mode():
                loss = diffusion_model(im, c)
            cum_loss += loss.item()
            batch_count += 1
        val_loss_tracker.append(cum_loss / batch_count)

        # Saving intermediate results
        plt.figure(figsize=(10,10))
        plt.plot(np.arange(0, ep+1), train_loss_tracker, label="Training loss")
        plt.plot(np.arange(0, ep+1, step=10), val_loss_tracker, label="Validation loss")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
        plt.close()
            
        if (cum_loss / batch_count) < min_val_loss:
            torch.save(diffusion_model.state_dict(), os.path.join(output_dir, f"model.pt"))
            min_val_loss = cum_loss / batch_count

print("TRAINING FINISHED!")