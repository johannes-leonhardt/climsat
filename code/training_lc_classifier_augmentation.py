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
import torchvision.transforms as T
from tqdm import tqdm

from dataset.dataset import Dataset
from models.lc_classifier import LCClassifier
from models.diffusion import GaussianDiffusionModel

# General settings
n_epoch = 20
batch_size = 128
n_T = 500
n_channels = 4
n_clim = 21
n_lc = 9
w = 0
lrate = 5e-5
country_train = "Finland"

# ClimSat augmentation settings
climsat_augmentation = True
run_name = "diffusion_2024-10-01_18-30-11_clim_lc"
if run_name is not None:
    diffusion_model = GaussianDiffusionModel(n_channels=n_channels, betas=(1e-4, 0.02), n_T=n_T, device=device, n_c=[n_clim+1, n_lc], normalization="MCBN")
    diffusion_model.load_state_dict(torch.load(os.path.join(os.path.join("..", "out", run_name, "model.pt"))))
    diffusion_model.eval()

# Color Jitter augmentation settings
# standard_augmentation = None
standard_augmentation = torch.nn.Sequential(
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
)

# Prepare output directory
run_name = f"lc_classification_{country_train}" + f"{'_climsat' if climsat_augmentation else ''}" + f"{'_cj' if standard_augmentation is not None else ''}" + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join("..", "out", run_name)
try:
    os.makedirs(output_dir)
except FileExistsError:
    print(f"{output_dir} already exists. Its contents may be overwritten.")

# Data stuff
data_root = os.path.join("..", "data")
regions = gpd.read_file(os.path.join(data_root, "lucas_regions.gpkg"))
regions_train = regions[regions.country == country_train]
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
classifier = LCClassifier(n_channels=n_channels, n_lc=n_lc).to(device)
optim = torch.optim.AdamW(classifier.parameters(), lr=lrate, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "max")

# Training
train_loss_tracker, val_loss_tracker = [], []
min_val_loss = np.inf

for ep in range(n_epoch):

    cum_loss = 0
    batch_count = 0

    # Training loop
    classifier.train()
    pbar = tqdm(train_dl, desc=f'Training epoch {ep}')
    for im, clim, lc, countries, _, filenames in pbar:
        optim.zero_grad()
        im, clim, lc = im.to(device), clim.to(device), lc.to(device)
        if climsat_augmentation:
            clim_t = clim[torch.randperm(im.shape[0])]
            c_enc = [clim, lc]
            c_dec = [clim_t, lc]
            with torch.inference_mode():
                z = diffusion_model.encode_ddim(x_i=im, c=c_enc, w=0, waitbar=True)
                im = diffusion_model.sample_ddim(code=z, c=c_dec, w=w, waitbar=True)
        if standard_augmentation is not None:
            im = torch.cat([standard_augmentation(im[:,[i]]) for i in range(im.shape[1])], dim=1)
        lc_hat = classifier(im.clone().to(device))
        loss = F.cross_entropy(lc_hat, torch.argmax(lc, dim=1))
        loss.backward()
        optim.step()
        cum_loss += loss.item()
        batch_count += 1
    train_loss_tracker.append(cum_loss / batch_count)
    
    # Validation loop
    classifier.eval()
    pbar = tqdm(val_dl, desc=f"Validation epoch {ep}")
    cum_loss = 0
    batch_count = 0
    for im, _, lc, _, _, _ in pbar:
        im, lc = im.to(device), lc.to(device)
        with torch.inference_mode():
            lc_hat = classifier(im)
            loss = F.cross_entropy(lc_hat, torch.argmax(lc, dim=1))
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
        torch.save(classifier.state_dict(), os.path.join(output_dir, f"model.pt"))
        min_val_loss = cum_loss / batch_count