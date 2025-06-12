import torch.nn as nn
import torch.nn.functional as F

from models.architectures import CNN, InvCNN, MLP

class FaderNetworkGAN(nn.Module):

    def __init__(self, n_channels, n_latent, n_c, normalization):

        super().__init__()

        self.encoder = CNN(n_channels, n_latent, n_c, normalization)
        self.decoder = InvCNN(n_latent, n_channels, n_c, normalization)
        if normalization == "CBN":
            self.clim_regressor = MLP(n_latent, n_c)
        elif normalization == "MCBN":
            self.clim_regressor = MLP(n_latent, n_c[0])
        self.discriminator = CNN(n_channels, 1)

    def forward(self, x, c):

        z = self.encoder(x, c)
        x = F.sigmoid(self.decoder(z, c))

        return x, z
    
    def encode(self, x, c):

        return self.encoder(x, c)
    
    def decode(self, z, c):

        return F.sigmoid(self.decoder(z, c))
    
    def regress_clim(self, z):

        return F.sigmoid(self.clim_regressor(z))
    
    def discriminate(self, x):

        return F.sigmoid(self.discriminator(x))