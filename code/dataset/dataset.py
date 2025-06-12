import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class Dataset(Dataset):

    img_mins = torch.tensor([0.0, 0.0, 0.0, 0.0]) / 10000
    img_maxs = torch.tensor([3000.0, 3000.0, 3000.0, 7000.0]) / 10000
    clim_mins = torch.tensor([1.5835872e+01, 6.4656720e-04, 0.0000000e+00, 1.3000032e+02, 1.0288679e+02, 1.8820821e+00, 2.4679015e+02, -7.9685547e+01, -4.3596970e-03, 0.0000000e+00, -4.3004095e+02, -1.1472540e+05, -2.3819749e+00, 2.0848077e+02, 5.2857483e+01, 3.1328849e-03, 0.0000000e+00, 2.4806209e+02, 3.3683371e+02, 4.2736177e+00, 2.7068048e+02])
    clim_maxs = torch.tensor([9.0658722e+01, 1.7434070e-02, 1.0531749e-04, 4.1450305e+02, 2.9235181e+02, 8.7800570e+00, 3.0482001e+02, 6.46447754e+01, 1.02141639e-02, 0.0000000e+00, 3.44702454e+02, 1.41503357e+02, 1.61314797e+00, 2.9431049e+02, 1.7917044e+02, 3.4271512e-02, 2.1449758e-03, 7.1001178e+02, 1.3091598e+05, 2.7111488e+01, 3.2113202e+02])

    def __init__(self, root, country, region):
        
        super().__init__()

        self.root = root
        self.country = country
        self.region = region

        self.image_path = os.path.join(root, "Images", "Sentinel-2", "2018", country)
        self.clim_path = os.path.join(root, "Climate", "CMIP6", "ssp245_2018", country)
        self.lc_path = os.path.join(root, "Land Cover", "ESAWC", "2020", country)
        self.filenames = [filename for filename in os.listdir(self.image_path) if (self.region in filename and filename.endswith(".pt"))]

    def __len__(self):

        return len(self.filenames)
    
    def __getitem__(self, idx):

        filename = self.filenames[idx]

        im = torch.load(os.path.join(self.image_path, filename))
        im = normalize_image(im, self.img_mins, self.img_maxs)
        clim = torch.load(os.path.join(self.clim_path, filename))
        clim = normalize_clim(clim, self.clim_mins, self.clim_maxs)
        lc = torch.load(os.path.join(self.lc_path, filename)).long()
        lc = remap_esawc(lc)

        return im, clim, lc, self.country, self.region, filename

def normalize_image(img, mins, maxs):

    return (img - mins[:, None, None]) / (maxs[:, None, None] - mins[:, None, None])

def normalize_clim(clim, mins, maxs):

    return (clim - mins) / (maxs - mins + 1e-6)

def unnormalize_clim(clim, mins, maxs):

    return clim * (maxs - mins + 1e-6) + mins

def remap_esawc(lc):

    lc[lc == 10] = 2 # Tree Cover
    lc[lc == 20] = 4 # Shrubland
    lc[lc == 30] = 3 # Grassland
    lc[lc == 40] = 1 # Cropland
    lc[lc == 50] = 0 # Built-up
    lc[lc == 60] = 5 # Bare/sparse vegetation
    lc[lc == 70] = 8 # Snow and Ice
    lc[lc == 80] = 6 # Permanent water bodies
    lc[lc == 90] = 7 # Herbaceous wetland
    lc[lc == 95] = 7 # Mangroves
    lc[lc == 100] = 5 # Moss and lichen
    lc = F.one_hot(lc, num_classes=9).squeeze().permute(2, 0, 1).float()

    return lc

def esawc_to_img(lc):

    colormap_esawc = [
        (0, np.array([192, 57, 43]) / 255), # Built-up
        (1, np.array([244, 208, 63]) / 255), # Cropland
        (2, np.array([11, 83, 69]) / 255), # Trees
        (3, np.array([121, 193, 113]) / 255), # Grassland
        (4, np.array([176, 102, 0]) / 255), # Shrubland
        (5, np.array([131, 145, 146]) / 255), # Bare / sparse vegetation
        (6, np.array([33, 97, 140]) / 255), # Permanent water bodies
        (7, np.array([174, 214, 241]) / 255), # Herbaceous wetland
        (8, np.array([255, 255, 255]) / 255), # Snow and ice
    ]

    lc_vis = np.zeros((lc.shape[0], lc.shape[1], 3))
    for i in range(len(colormap_esawc)):
        lc_vis[lc.numpy() == colormap_esawc[i][0]] = colormap_esawc[i][1]
    
    return lc_vis
