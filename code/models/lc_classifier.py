import torch.nn as nn

from models.architectures import UNet

class LCClassifier(nn.Module):

    def __init__(self, n_channels, n_lc):

        super().__init__()

        self.regressor = UNet(n_channels, n_lc)

    def forward(self, x):

        return self.regressor(x)
