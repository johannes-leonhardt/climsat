import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda:0"

import geopandas as gpd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import Dataset
from models.autoencoder import Autoencoder
from models.aae import AAE
from models.fader_network import FaderNetwork
from models.fader_network_gan import FaderNetworkGAN
from models.aegan import AEGAN

# General settings
model_class = Autoencoder
run_name = "ae_2024-10-01_18-27-43"
# model_class = AAE
# run_name = "aae_2024-11-13_15-36-44"
# model_class = AEGAN
# run_name = "aegan_2024-11-13_15-51-12"
# model_class = FaderNetwork
# run_name = "fader_2024-11-13_16-24-51"
# model_class = FaderNetworkGAN
# run_name = "fader_gan_2024-11-19_11-43-27"
batch_size = 512
n_channels = 4
n_clim = 21
n_lc = 9
n_latent = 1024

# Data stuff
data_root = os.path.join("..", "data")
gen_dir = os.path.join(data_root, "Experiments", "Editing", f"{run_name}_clim")
regions = gpd.read_file(os.path.join(data_root, "lucas_regions.gpkg"))
regions_test = regions[regions.split == "test"]

# Model stuff
model_path = os.path.join("..", "out", run_name, "model.pt")
model = model_class(n_channels=n_channels, n_latent=n_latent, n_c=[n_clim, n_lc], normalization="MCBN").to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Run inference
for country, region in zip(regions_test.country, regions_test.region):
    gen_dir_im_i = os.path.join(gen_dir, country)
    try:
        os.makedirs(gen_dir_im_i)
    except FileExistsError:
        pass
    permutations_dir = os.path.join(data_root, "Climate", "CMIP6", "ssp245_2018_permuted")
    ds = Dataset(data_root, country, region)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
    pbar = tqdm(dl, desc=f"Running inference for {country}, {region}")
    for im, clim_s, lc_s, _, _, filenames in pbar:
        im, clim_s, lc_s = im.to(device), clim_s.to(device), lc_s.to(device)
        # Sample conditions for transfer
        clim_t = torch.stack([torch.load(os.path.join(permutations_dir, country, filename)) for filename in filenames], dim=0).to(device)
        c_enc = [clim_s, lc_s]
        c_dec = [clim_t, lc_s]
        with torch.inference_mode():
            # Encode
            z_hat = model.encode(im, c_enc)
            # Decode
            im_hat = model.decode(z_hat, c_dec)
        # Save
        for i in range(im_hat.shape[0]):
            torch.save(im_hat[i].cpu().clone(), os.path.join(gen_dir_im_i, filenames[i]))