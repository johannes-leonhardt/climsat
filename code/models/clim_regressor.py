import torch.nn as nn
import torch.nn.functional as F

from models.architectures import CNN

class ClimRegressor(nn.Module):

    def __init__(self, n_channels, n_clim):

        super().__init__()

        self.regressor = CNN(n_channels, n_clim)

    def forward(self, x):

        return F.sigmoid(self.regressor(x))

