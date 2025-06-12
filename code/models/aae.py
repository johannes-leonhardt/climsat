import torch.nn as nn
import torch.nn.functional as F

from models.architectures import CNN, InvCNN, MLP

class AAE(nn.Module):

    def __init__(self, n_channels, n_latent, n_c, normalization):

        super().__init__()
        self.encoder = CNN(n_channels, n_latent, n_c, normalization)
        self.decoder = InvCNN(n_latent, n_channels, n_c, normalization)
        self.discriminator = MLP(n_latent, 1)

    def forward(self, x, c=None):

        z = self.encoder(x, c)
        x = F.sigmoid(self.decoder(z, c))

        return x, z
    
    def encode(self, x, c=None):

        return self.encoder(x, c)
    
    def decode(self, z, c=None):

        return F.sigmoid(self.decoder(z, c))
    
    def discriminate(self, z):

        return F.sigmoid(self.discriminator(z))
