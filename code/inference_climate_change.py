import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda:0"
import math

import numpy as np
import rasterio as rio
import torch
import torch.nn.functional as F

from models.diffusion import GaussianDiffusionModel
from dataset.dataset import Dataset, normalize_image, normalize_clim, remap_esawc

# General settings
run_name = "diffusion_2024-10-01_18-30-11_clim_lc"
n_T = 500
n_channels = 4
n_clim = 21
n_lc = 9
w = 0
protected_area_name = "Vesuvio"

# Data stuff
data_dir = os.path.join("..", "data", "Experiments", "Climate Change", protected_area_name)
img = rio.open(os.path.join(data_dir, "img", f"{protected_area_name}.tif")).read()
clim_245 = np.load(os.path.join(data_dir, "clim", f"{protected_area_name}_ssp245.npy"))
clim_585 = np.load(os.path.join(data_dir, "clim", f"{protected_area_name}_ssp585.npy"))
lc = rio.open(os.path.join(data_dir, "lc", f"{protected_area_name}.tif")).read()
gen_dir = os.path.join(data_dir, "edit")
try:
    os.makedirs(gen_dir)
except FileExistsError:
    pass
img = normalize_image(torch.from_numpy(img).float(), Dataset.img_mins*10000, Dataset.img_maxs*10000).to(device)[:, :512, :512]
clim_245 = [normalize_clim(torch.from_numpy(clim), Dataset.clim_mins, Dataset.clim_maxs).to(device) for clim in clim_245]
clim_585 = [normalize_clim(torch.from_numpy(clim), Dataset.clim_mins, Dataset.clim_maxs).to(device) for clim in clim_585]
lc = remap_esawc(torch.from_numpy(lc).long().squeeze()).to(device)[:, :512, :512]
no_val = torch.all(img == 0, dim=0)
original_shape = [img.shape[1], img.shape[2]]
resized_shape = [2 ** math.ceil(math.log2(img.shape[1])), 2 ** math.ceil(math.log2(img.shape[2]))]
img = F.pad(img, (0, resized_shape[1] - original_shape[1], 0, resized_shape[0]- original_shape[0]))
lc = F.pad(lc, (0, resized_shape[1] - original_shape[1], 0, resized_shape[0] - original_shape[0]))

# Model stuff
model_dir = os.path.join("..", "out", run_name, "model.pt")
diffusion_model = GaussianDiffusionModel(n_channels=n_channels, betas=(1e-4, 0.02), n_T=n_T, device=device, n_c=[n_clim+1, n_lc], normalization="MCBN")
diffusion_model.load_state_dict(torch.load(os.path.join(model_dir), weights_only=True))
diffusion_model.eval()

# Encode image
c_enc = [clim_245[0].unsqueeze(0), lc.unsqueeze(0)]
with torch.inference_mode():
    z = diffusion_model.encode_ddim(
        x_i = img.unsqueeze(0),
        c = c_enc,
        w = 0,
        waitbar=True
    )

# Run model with scenarios
for path in ["ssp245", "ssp585"]:
    if path == "ssp245":
        clim_ts = clim_245
    elif path == "ssp585":
        clim_ts = clim_585
    for i in range(0, len(clim_ts)):
        print(f"Processing {path}, {i + 2018}...")
        c_dec = [clim_ts[i].unsqueeze(0), lc.unsqueeze(0)]
        with torch.inference_mode():
            im_hat = diffusion_model.sample_ddim(
                code = z,
                c = c_dec,
                w = w,
                waitbar=True
            )
            im_hat = im_hat[0, :, :original_shape[0], :original_shape[1]]
            im_hat[no_val[None, :, :].expand(4, -1, -1)] = 0

        torch.save(im_hat.cpu().clone(), os.path.join(gen_dir, f"{path}_{i + 2018}.pt"))