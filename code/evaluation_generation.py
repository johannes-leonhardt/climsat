import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
device = "cuda:0"

import geopandas as gpd
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics.classification import Accuracy
from tqdm import tqdm

from dataset.dataset import Dataset
from models.clim_regressor import ClimRegressor
from models.lc_classifier import LCClassifier

# General settings
# run_name = None # for evaluating the real images
# run_name = "ae_2024-10-01_18-27-43"
# run_name = "aae_2024-11-13_15-36-44"
# run_name = "aegan_2024-11-13_15-51-12"
# run_name = "fader_2024-11-13_16-24-51"
# run_name = "fader_gan_2024-11-14_13-33-14"
# run_name = "diffusion_2024-10-05_18-38-46_clim"
run_name = "diffusion_2024-10-01_18-30-11_clim_lc"
batch_size = 512
n_channels = 4
n_clim = 21
n_lc = 9
ws = [0, 0.2, 0.4, 0.6, 0.8, 1] # For models other than diffusion, set to ws = [None]

# Data stuff
data_root = os.path.join("..", "data")
regions = gpd.read_file(os.path.join(data_root, "lucas_regions.gpkg"))
regions_val = regions[regions.split == "val"]
regions_test = regions[regions.split == "test"]

# Evaluation
for w in ws:

    model_dir = os.path.join("..", "out", run_name if run_name is not None else "real")
    if run_name is not None:
        run_name_i = f"{run_name}{'_'+str(w) if w is not None else ''}_clim"
    else: 
        run_name_i = run_name

    print(f"Evaluating generated images from {run_name_i}")

    # Image quality metrics
    inception_rgb = InceptionScore(normalize=True).to(device)
    inception_nirrg = InceptionScore(normalize=True).to(device)
    fid_rgb = FrechetInceptionDistance(normalize=True).to(device)
    fid_nirrg = FrechetInceptionDistance(normalize=True).to(device)
    kid_rgb = KernelInceptionDistance(normalize=True).to(device)
    kid_nirrg = KernelInceptionDistance(normalize=True).to(device)

    # Assess image quality of generated data
    for country, region in zip(regions_test.country, regions_test.region):
        ds = Dataset(data_root, country, region)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
        if run_name is not None:
            im_dir = os.path.join(data_root, "Generated", f"{run_name_i}", country)
        for im, _, _, _, _, filenames in tqdm(dl, desc=f"Evaluating image quality for {country}, {region}"):
            if run_name is not None:
                im = torch.stack([torch.load(os.path.join(im_dir, filename)) for filename in filenames])
            im = im.to(device)
            # Update image quality metrics
            inception_rgb.update(im[:,[2,1,0]])
            inception_nirrg.update(im[:,[3,2,1]])
            fid_rgb.update(im[:,[2,1,0]], real=False)
            fid_nirrg.update(im[:,[3,2,1]], real=False)
            kid_rgb.update(im[:,[2,1,0]], real=False)
            kid_nirrg.update(im[:,[3,2,1]], real=False)

    # Samples are compared to validation data for assessing image quality
    for country, region in zip(regions_val.country, regions_val.region):
        ds = Dataset(data_root, country, region)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
        for im, _, _, _, _, _ in tqdm(dl, desc=f"Compare to real images of {country}, {region}"):
            im = im.to(device)
            fid_rgb.update(im[:,[2,1,0]], real=True)
            fid_nirrg.update(im[:,[3,2,1]], real=True)
            kid_rgb.update(im[:,[2,1,0]], real=True)
            kid_nirrg.update(im[:,[3,2,1]], real=True)

    # Compute and save
    inception_rgb_result = inception_rgb.compute()[0].item()
    inception_nirrg_result = inception_nirrg.compute()[0].item()
    fid_rgb_result = fid_rgb.compute().item()
    fid_nirrg_result = fid_nirrg.compute().item()
    kid_rgb_result = kid_rgb.compute()[0].item()
    kid_nirrg_result = kid_nirrg.compute()[0].item()
    with open(os.path.join(model_dir, f"metrics_{run_name_i}.txt"), "w") as output:
        output.write("Image quality metrics\n---\n")
        output.write(f"Inception score (RGB): {inception_rgb_result}\n")
        output.write(f"Inception score (NIRRG): {inception_nirrg_result}\n")
        output.write(f"FID (RGB): {fid_rgb_result}\n")
        output.write(f"FID (NIRRG): {fid_nirrg_result}\n")
        output.write(f"KID (RGB): {kid_rgb_result}\n")
        output.write(f"KID (NIRRG): {kid_nirrg_result}\n")
    del inception_rgb, inception_nirrg, fid_rgb, fid_nirrg

    # FF to c_clim
    cc = PearsonCorrCoef(num_outputs=n_clim).to(device)

    # Load regression model
    run_name_regression = "clim_regression_2024-10-01_18-07-45"
    clim_regression_dir = os.path.join("..", "out", run_name_regression)
    clim_regressor = ClimRegressor(n_channels=n_channels, n_clim=n_clim).to(device)
    clim_regressor.load_state_dict(
        torch.load(os.path.join(clim_regression_dir, "model.pt")),
        strict=True   
    )
    clim_regressor.eval()

    # Assess climate regression of generated data
    for country, region in zip(regions_test.country, regions_test.region):
        ds = Dataset(data_root, country, region)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
        if run_name is not None:
            im_dir = os.path.join(data_root, "Generated", f"{run_name_i}", country)
        for im, clim, _, _, _, filenames in tqdm(dl, desc=f"Evaluating climate regressor for {country}, {region}"):
            if run_name is not None:
                im = torch.stack([torch.load(os.path.join(im_dir, filename)) for filename in filenames])
            im, clim = im.to(device), clim.to(device)
            with torch.inference_mode():
                clim_hat = clim_regressor(im)
            clim, clim_hat = clim[~torch.any(torch.isnan(clim_hat), dim=1)], clim_hat[~torch.any(torch.isnan(clim_hat), dim=1)]
            cc.update(clim_hat, clim)

    # Compute and save
    cc_result = torch.nanmean(cc.compute()).item()
    with open(os.path.join(model_dir, f"metrics_{run_name_i}.txt"), "a") as output:
        output.write("\nClimate regression metrics\n---\n")
        output.write(f"Correlation coefficient: {cc_result}\n")
    del cc

    # FF to c_LC
    acc = Accuracy(task="multiclass", num_classes=n_lc).to(device)

    # Load classification model
    run_name_classification = "lc_classification_all_2024-11-27_00-03-14"
    lc_classification_dir = os.path.join("..", "out", run_name_classification)
    lc_classifier = LCClassifier(n_channels=n_channels, n_lc=n_lc).to(device)
    lc_classifier.load_state_dict(
        torch.load(os.path.join(lc_classification_dir, "model.pt")),
        strict=True
    )
    lc_classifier.eval()

    # Assess land cover classification of generated data
    for country, region in zip(regions_test.country, regions_test.region):
        ds = Dataset(data_root, country, region)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
        if run_name is not None:
            im_dir = os.path.join(data_root, "Generated", f"{run_name_i}", country)
        for im, _, lc, _, _, filenames in tqdm(dl, desc=f"Evaluating land cover classifier for {country}, {region}"):
            if run_name is not None:
                im = torch.stack([torch.load(os.path.join(im_dir, filename)) for filename in filenames])
            lc, im = lc.to(device), im.to(device)
            with torch.inference_mode():
                lc_hat = lc_classifier(im)
            acc.update(torch.argmax(lc_hat, dim=1), torch.argmax(lc, dim=1))

    # Compute and save
    acc_result = acc.compute().item()
    with open(os.path.join(model_dir, f"metrics_{run_name_i}.txt"), "a") as output:
        output.write("\nLand cover classification metrics\n---\n")
        output.write(f"Overall accuracy: {acc_result}\n")
    del acc