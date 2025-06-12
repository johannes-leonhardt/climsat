import torch.nn as nn
import torch.nn.functional as F

from models.architectures import CNN, InvCNN, MLP

class FaderNetwork(nn.Module):

    def __init__(self, n_channels, n_latent, n_c, normalization):

        super().__init__()

        self.encoder = CNN(n_channels, n_latent, n_c, normalization)
        self.decoder = InvCNN(n_latent, n_channels, n_c, normalization)
        if normalization == "CBN":
            self.clim_regressor = MLP(n_latent, n_c)
        elif normalization == "MCBN":
            self.clim_regressor = MLP(n_latent, n_c[0])

    def forward(self, x, c=None):

        z = self.encoder(x, c)
        x = F.sigmoid(self.decoder(z, c))

        return x, z
    
    def encode(self, x, c=None):

        return self.encoder(x, c)
    
    def decode(self, z, c=None):

        return F.sigmoid(self.decoder(z, c))
    
    def regress_clim(self, z):

        return F.sigmoid(self.clim_regressor(z))