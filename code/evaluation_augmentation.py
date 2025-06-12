import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda:0"

import geopandas as gpd
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
from tqdm import tqdm

from dataset.dataset import Dataset
from models.lc_classifier import LCClassifier

# General settings
run_name = "lc_classification_Finland_2024-10-09_15-58-16"
batch_size = 512
n_channels = 4
n_lc = 9

# Data stuff
data_root = os.path.join("..", "data")
regions = gpd.read_file(os.path.join(data_root, "lucas_regions.gpkg"))
regions_test = regions[regions.split == "test"]

# Model stuff
model_dir = os.path.join("..", "out", run_name)
lc_classifier = LCClassifier(n_channels=n_channels, n_lc=n_lc).to(device)
lc_classifier.load_state_dict(
    torch.load(os.path.join(model_dir, "model.pt")),
    strict=True
)
lc_classifier.eval()

# Initialize metric
acc = Accuracy(task="multiclass", num_classes=n_lc).to(device)

# Assess land cover classification of generated data
for country, region in zip(regions_test.country, regions_test.region):
    ds = Dataset(data_root, country, region)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
    for im, _, lc, _, _, _ in tqdm(dl, desc=f"Evaluating land cover classifier for {country}, {region}"):
        lc, im = lc.to(device), im.to(device)
        with torch.inference_mode():
            lc_hat = lc_classifier(im)
        acc.update(torch.argmax(lc_hat, dim=1), torch.argmax(lc, dim=1))

# Compute and save
acc_result = acc.compute().item()
with open(os.path.join(model_dir, f"metrics.txt"), "a") as output:
    output.write("\nLand cover classification metrics\n---\n")
    output.write(f"Overall accuracy: {acc_result}\n")
del acc