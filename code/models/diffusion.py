import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from models.architectures import UNet

def dm_schedules(beta1, beta2, T):

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = 1 - beta_t
    alphabar_t = torch.cumsum(torch.log(alpha_t), dim=0).exp()
    alphabar_t_minus_1 = F.pad(alphabar_t[:-1], pad=(1, 0), value=1.)
    alphabar_t_plus_1 = F.pad(alphabar_t[1:], pad=(0, 1), value=0.)

    return {
        "beta_t": beta_t,
        "alpha_t": alpha_t,
        "alphabar_t": alphabar_t,
        "alphabar_t_minus_1": alphabar_t_minus_1,
        "alphabar_t_plus_1": alphabar_t_plus_1
    }
    
class GaussianDiffusionModel(nn.Module):
    
    def __init__(self, n_channels, betas, n_T, device, n_c=None, normalization="BN"):

        super().__init__()
        
        self.nn_model = UNet(n_in=n_channels, n_out=n_channels, n_c=n_c, normalization=normalization)
        self.n_T = n_T
        self.device = device
        for k, v in dm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
        self.to(device)

    def _prepare_conditions(self, t, c=None, p_uncond=0):

        t = t / self.n_T
        if c is None:
            tc = t[:,None]
        elif torch.is_tensor(c):
            mask = torch.bernoulli((1 - p_uncond) * torch.ones((c.shape[0]))).to(self.device)
            if c.dim() == 2:
                c = mask[:, None] * c
                tc = torch.concatenate([c, t[:,None]], dim=1)
            elif c.dim() == 4:
                c = mask[:, None, None, None] * c
                tc = [t[:, None], c]
        elif isinstance(c, list):
            mask = torch.bernoulli((1 - p_uncond) * torch.ones((c[0].shape[0]))).to(self.device)
            c = [mask[:,None] * c[0], mask[:, None, None, None] * c[1]]
            tc = [torch.concatenate([c[0], t[:, None]], dim=1), c[1]]
        
        return tc

    def forward(self, x, c=None, p_uncond=0):

        t = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device) # t ~ Uniform(0, n_T)
        c = self._prepare_conditions(t, c, p_uncond)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        x_t = (
            torch.sqrt(self.alphabar_t[t, None, None, None]) * x
            + torch.sqrt(1 - self.alphabar_t[t, None, None, None]) * noise
        )
        out = self.nn_model(x_t, c)
        loss = F.mse_loss(noise, out)
        
        return loss
    
    def sample_ddim(self, n_sample=None, size=None, code=None, c=None, w=0, waitbar=False):

        if n_sample is None:
            n_sample = code.shape[0]
            size = code.shape[1:]
        x_i = torch.randn(n_sample, *size).to(self.device) if code is None else code 
        n_T_seq = tqdm(range(self.n_T, 0, -1)) if waitbar else range(self.n_T, 0, -1)
        for i in n_T_seq:
            t = torch.tensor([i]).repeat(n_sample).to(self.device)
            tc_cond = self._prepare_conditions(t, c, p_uncond=0)
            tc_uncond = self._prepare_conditions(t, c, p_uncond=1)
            eps_cond = self.nn_model(x_i, tc_cond)
            eps_uncond = self.nn_model(x_i, tc_uncond)
            eps = (1 + w) * eps_cond - w * eps_uncond
            x_i = (
                (torch.sqrt(self.alphabar_t_minus_1[i]) / torch.sqrt(self.alphabar_t[i]))
                * (x_i - torch.sqrt(1 - self.alphabar_t[i]) * eps)
                + torch.sqrt(1 - self.alphabar_t_minus_1[i]) * eps
            )

        return x_i
    
    def encode_ddim(self, x_i, c=None, w=0, waitbar=False):

        n_T_seq = tqdm(range(self.n_T)) if waitbar else range(self.n_T)
        for i in n_T_seq:
            t = torch.tensor([i]).repeat(x_i.shape[0]).to(self.device)
            tc_cond = self._prepare_conditions(t, c, p_uncond=0)
            tc_uncond = self._prepare_conditions(t, c, p_uncond=1)
            eps_cond = self.nn_model(x_i, tc_cond)
            eps_uncond = self.nn_model(x_i, tc_uncond)
            eps = (1 + w) * eps_cond - w * eps_uncond
            x_i = (
                (torch.sqrt(self.alphabar_t_plus_1[i]) / torch.sqrt(self.alphabar_t[i]))
                * (x_i - torch.sqrt(1 - self.alphabar_t[i]) * eps)
                + torch.sqrt(1 - self.alphabar_t_plus_1[i]) * eps
            )

        return x_i