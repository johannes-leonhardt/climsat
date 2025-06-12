import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda:0"

import geopandas as gpd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.diffusion import GaussianDiffusionModel
from dataset.dataset import Dataset

# General settings
# run_name = "diffusion_2024-10-05_18-38-46_clim"
run_name = "diffusion_2024-10-01_18-30-11_clim_lc"
batch_size = 512
n_T = 500
n_channels = 4
n_clim = 21
use_lc = True
n_lc = 9
w = 0
model_path = os.path.join("..", "out", run_name, "model.pt")

# Data stuff
data_root = os.path.join("..", "data")
gen_dir = os.path.join(data_root, "Experiments", "Editing", f"{run_name}_{w}_clim")
regions = gpd.read_file(os.path.join(data_root, "lucas_regions.gpkg"))
regions_test = regions[regions.split == "test"] # full run

# Model stuff
if not use_lc:
    n_c = n_clim + 1
    normalization = "CBN"
else:
    n_c = [n_clim + 1, n_lc]
    normalization = "MCBN"
diffusion_model = GaussianDiffusionModel(n_channels=n_channels, betas=(1e-4, 0.02), n_T=n_T, device=device, n_c=n_c, normalization=normalization)
diffusion_model.load_state_dict(torch.load(os.path.join(model_path)))
diffusion_model.eval()

# Run inference
for country, region in zip(regions_test.country, regions_test.region):
    gen_dir_im_i = os.path.join(gen_dir, country)
    try:
        os.makedirs(gen_dir_im_i)
    except FileExistsError:
        pass
    derang_dir_clim = os.path.join(data_root, "Climate", "ssp245_2018_permutations")
    ds = Dataset(data_root, country, region)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
    pbar = tqdm(dl, desc=f"Running inference for {country}, {region}")
    for im, clim_s, lc_s, _, _, filenames in pbar:
        im, clim_s, lc_s = im.to(device), clim_s.to(device), lc_s.to(device)
        # Load conditions for transfer
        clim_t = torch.stack([torch.load(os.path.join(derang_dir_clim, country, filename)) for filename in filenames], dim=0).to(device)
        if not use_lc:
            c_enc = clim_s
            c_dec = clim_t
        else:
            c_enc = [clim_s, lc_s]
            c_dec = [clim_t, lc_s]
        with torch.inference_mode():
            # Encode
            code = diffusion_model.encode_ddim(
                x_i = im,
                c = c_enc,
                w = 0
            )
            # Decode
            im_hat = diffusion_model.sample_ddim(
                code = code,
                c = c_dec,
                w = w
            )
        # Save
        for i in range(im_hat.shape[0]):
            torch.save(im_hat[i].cpu().clone(), os.path.join(gen_dir_im_i, filenames[i]))