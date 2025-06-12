import torch.nn as nn
import torch.nn.functional as F

class ConditionalBatchNorm2d(nn.Module):
  
    def __init__(self, n, n_c):

        super().__init__()
        self.n = n
        self.bn = nn.BatchNorm2d(n, affine=False)
        self.linear = nn.Linear(n_c, n * 2)

    def forward(self, x, c):

        x = self.bn(x)
        gamma, beta = self.linear(c).chunk(2, 1)
        x = gamma.view(-1, self.n, 1, 1) * x + beta.view(-1, self.n, 1, 1)
        
        return x

class MultiConditionalBatchNorm2d(nn.Module):
  
    def __init__(self, n, n_c):

        super().__init__()
        self.n = n
        self.bn = nn.BatchNorm2d(n, affine=False)
        self.linear_all_beta = nn.Linear(n_c[0], n)
        self.linear_px_beta = nn.Conv2d(n_c[1], n, kernel_size=1)
        self.linear_all_gamma = nn.Linear(n_c[0], n)
        self.linear_px_gamma = nn.Conv2d(n_c[1], n, kernel_size=1)

    def forward(self, x, c):

        x = self.bn(x)
        c0 = c[0]
        c1 = c[1]
        c1 = F.avg_pool2d(c1, kernel_size=c1.shape[2]//x.shape[2])
        gamma_all = self.linear_all_gamma(c0).view(-1, self.n, 1, 1)
        beta_all = self.linear_all_beta(c0).view(-1, self.n, 1, 1)
        gamma_px = self.linear_px_gamma(c1)
        beta_px = self.linear_px_beta(c1)
        gamma = gamma_all * gamma_px
        beta = beta_all + beta_px
        x = gamma * x + beta
        
        return x