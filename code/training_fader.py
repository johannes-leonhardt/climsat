import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from dataset.dataset import Dataset
from models.fader_network import FaderNetwork

# General settings
n_epoch = 500
batch_size = 128
device = "cuda:0"
n_channels = 4
n_clim = 21
n_lc = 9
n_latent = 1024
lrate = 5e-5

# Prepare output directory
run_name = "fader_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
model = FaderNetwork(n_channels=n_channels, n_latent=n_latent, n_c=[n_clim, n_lc], normalization="MCBN").to(device)
optim_reconstruction = torch.optim.Adam(nn.ModuleList((model.encoder, model.decoder)).parameters(), lr=lrate)
optim_attribute = torch.optim.Adam(model.encoder.parameters(), lr=lrate)
optim_regressor = torch.optim.Adam(model.clim_regressor.parameters(), lr=lrate)

# Training
train_loss_tracker, val_loss_tracker = [], []
min_val_loss = np.inf

for ep in range(n_epoch+1):

    optim_reconstruction.param_groups[0]['lr'] = lrate * (1 - ep/n_epoch)
    optim_attribute.param_groups[0]['lr'] = lrate * (1 - ep/n_epoch)
    optim_regressor.param_groups[0]['lr'] = lrate * (1 - ep/n_epoch)

    # Training loop
    for im, clim, lc, _, _, _ in tqdm(train_dl, desc=f"Training epoch {ep}"):
        
        im, clim, lc = im.to(device), clim.to(device), lc.to(device)
        c = [clim, lc]
        
        # Reconstruction loss
        model.train()
        optim_reconstruction.zero_grad()
        im_hat, _ = model(im, c)
        reconstruction_loss = F.mse_loss(im_hat, im)
        reconstruction_loss.backward()
        optim_reconstruction.step()
        
        # Encoder loss
        optim_attribute.zero_grad()
        z_fake = model.encode(im, c)
        clim_hat = model.regress_clim(z_fake)
        attribute_loss = 0.1 * -F.mse_loss(clim_hat, clim)
        attribute_loss.backward()
        optim_attribute.step()
        
        # Regressor loss
        optim_regressor.zero_grad()
        z_fake = model.encode(im, c)
        clim_hat = model.regress_clim(z_fake)
        regression_loss = 0.1 * F.mse_loss(clim_hat, clim)
        regression_loss.backward()
        optim_regressor.step()
        
    # Validation loop
    if ep % 10 == 0:
        model.eval()
        train_loss, val_loss = 0, 0
        train_batch_count, val_batch_count = 0, 0
        for im, clim, lc, _, _, _ in tqdm(train_dl, desc=f"Evaluation on training data..."):
            im, clim, lc = im.to(device), clim.to(device), lc.to(device)
            c = [clim, lc]
            with torch.inference_mode():
                im_hat, _ = model(im, c)
                loss = F.mse_loss(im_hat, im)
            train_loss += loss.item()
            train_batch_count += 1
        train_loss_tracker.append(train_loss / train_batch_count)
        for im, clim, lc, _, _, _ in tqdm(val_dl, desc=f"Evaluation on validation data..."):
            im, clim, lc = im.to(device), clim.to(device), lc.to(device)
            c = [clim, lc]
            with torch.inference_mode():
                im_hat, _ = model(im, c)
                loss = F.mse_loss(im_hat, im)
            val_loss += loss.item()
            val_batch_count += 1
        val_loss_tracker.append(val_loss / val_batch_count)

        # Saving intermediate results
        plt.figure(figsize=(10,10))
        plt.plot(np.arange(0, ep+1, step=10), train_loss_tracker, label="Training loss")
        plt.plot(np.arange(0, ep+1, step=10), val_loss_tracker, label="Validation loss")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
        plt.close()
        if (val_loss / val_batch_count) < min_val_loss:
            torch.save(model.state_dict(), os.path.join(output_dir, f"model.pt"))
            min_val_loss = val_loss / val_batch_count

print("TRAINING FINISHED!")