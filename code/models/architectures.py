### For architectures which can be used in a variety of settings

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cbn import ConditionalBatchNorm2d, MultiConditionalBatchNorm2d

### U-NET ###

class UNet(nn.Module):

    def __init__(self, n_in, n_out, n_c=None, normalization="BN"):

        # Normalization in "BN", "CBN", "MCBN"

        super().__init__()
        self.inc = DoubleConv(n_in, 64, n_c, normalization)
        self.down1 = Down(64, 128, n_c, normalization)
        self.down2 = Down(128, 256, n_c, normalization)
        self.down3 = Down(256, 512, n_c, normalization)
        self.up1 = UpWithSkip(512, 256, n_c, normalization)
        self.up2 = UpWithSkip(256, 128, n_c, normalization)
        self.up3 = UpWithSkip(128, 64, n_c, normalization)
        self.out = nn.Conv2d(64, n_out, kernel_size=1)

    def forward(self, x, c=None):

        x1 = self.inc(x, c)
        x2 = self.down1(x1, c)
        x3 = self.down2(x2, c)
        x4 = self.down3(x3, c)
        x = self.up1(x4, x3, c)
        x = self.up2(x, x2, c)
        x = self.up3(x, x1, c)
        x = self.out(x)

        return x
    
### CNN ###
    
class CNN(nn.Module):

    def __init__(self, n_in, n_out, n_c=None, normalization="BN"):

        super().__init__()
        self.inc = DoubleConv(n_in, 64, n_c, normalization)
        self.down1 = Down(64, 128, n_c, normalization)
        self.down2 = Down(128, 256, n_c, normalization)
        self.down3 = Down(256, 512, n_c, normalization)
        self.out = nn.Linear(512 * 8 * 8, n_out)

    def forward(self, x, c=None):
        x = self.inc(x, c)
        x = self.down1(x, c)
        x = self.down2(x, c)
        x = self.down3(x, c)
        x = x.reshape(x.shape[0], -1)
        x = self.out(x)

        return x
    
### Inverted CNN ###
    
class InvCNN(nn.Module):

    def __init__(self, n_in, n_out, n_c=None, normalization="BN"):

        super().__init__()

        self.bottleneck_out = nn.Linear(n_in, 512 * 8 * 8)
        self.up1 = Up(512, 256, n_c, normalization)
        self.up2 = Up(256, 128, n_c, normalization)
        self.up3 = Up(128, 64, n_c, normalization)
        self.out = nn.Conv2d(64, n_out, kernel_size=1)

    def forward(self, x, c=None):
        x = self.bottleneck_out(x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1, 8, 8)
        x = self.up1(x, c)
        x = self.up2(x, c)
        x = self.up3(x, c)
        x = self.out(x)

        return x

### MLP ###

class MLP(nn.Module):

    def __init__(self, n_in, n_out):

        super().__init__()
        self.layer1 = nn.Linear(n_in, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, n_out)

    def forward(self, x):

        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.out(x)

        return x
    
### Building blocks ###
    
class DoubleConv(nn.Module):

    def __init__(self, n_in, n_out, n_c=None, normalization="BN"):

        super().__init__()
        self.normalization = normalization
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1, bias=False)
        if normalization == "BN":
            self.bn1 = nn.BatchNorm2d(n_out)
            self.bn2 = nn.BatchNorm2d(n_out)
        elif normalization == "CBN":
            self.bn1 = ConditionalBatchNorm2d(n_out, n_c)
            self.bn2 = ConditionalBatchNorm2d(n_out, n_c)
        elif normalization == "MCBN":
            self.bn1 = MultiConditionalBatchNorm2d(n_out, n_c)
            self.bn2 = MultiConditionalBatchNorm2d(n_out, n_c)

    def forward(self, x, c=None):

        x = self.conv1(x)
        if self.normalization == "BN":
            x = self.bn1(x)
        else:
            x = self.bn1(x, c)
        x = F.relu(x)
        x = self.conv2(x)
        if self.normalization == "BN":
            x = self.bn2(x)
        else:
            x = self.bn2(x, c)
        x = F.relu(x)

        return x

class Down(nn.Module):

    def __init__(self, n_in, n_out, n_c=None, normalization="BN"):

        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.double_conv = DoubleConv(n_in, n_out, n_c, normalization)

    def forward(self, x, c=None):
        
        x = self.pool(x)
        x = self.double_conv(x, c)

        return x

class UpWithSkip(nn.Module):

    def __init__(self, n_in, n_out, n_c=None, normalization="BN"):

        super().__init__()
        self.linear = nn.Conv2d(n_in, n_in // 2, kernel_size=1, stride=1)
        self.double_conv = DoubleConv(n_in, n_out, n_c, normalization)

    def forward(self, x, x_res, c=None):

        x = self.linear(x)
        x_up = F.interpolate(x, scale_factor=2)
        x = torch.cat([x_res, x_up], dim=1)
        x = self.double_conv(x, c)

        return x
    
class Up(nn.Module):

    def __init__(self, n_in, n_out, n_c=None, normalization="BN"):

        super().__init__()
        self.double_conv = DoubleConv(n_in, n_out, n_c, normalization)

    def forward(self, x, c=None):

        x = F.interpolate(x, scale_factor=2)
        x = self.double_conv(x, c)

        return x