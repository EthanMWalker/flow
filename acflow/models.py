import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from acflow.coupling import AffineCoupling

import numpy as np
LOG_2_PI = np.log(2*np.pi)


class RealNVP(nn.Module):

  def __init__(self, in_channels, mid_channels, num_layers, n_comps, 
              shape, device, num_blocks=6):
    super().__init__()

    self.num_layers = num_layers
    self.n_comps = n_comps
    self.gmm_dim = np.prod(shape)
    self.shape = shape
    self.device = device

    self.flows = nn.ModuleList(
      [
        AffineCoupling(in_channels, mid_channels, num_blocks) 
        for _ in range(num_layers)
      ]
    )

    self.means = nn.Parameter(torch.randn(n_comps, self.gmm_dim).to(device))
    self.covs = nn.Parameter(torch.rand(n_comps, self.gmm_dim).to(device))
    # self.means = torch.randn(n_comps, self.gmm_dim).to(device)
    # self.covs = torch.rand(n_comps, self.gmm_dim).to(device)
    self.mix = D.Categorical(torch.ones(self.n_comps,).to(self.device))
    self.prior = self.update_gmm()

    self.register_buffer('data_constraint', torch.tensor([0.9]))

  @property
  def mask(self):
    temp = torch.arange(self.shape[0]*self.shape[1]*self.shape[2])
    zero_mask = temp%2 == 0
    return zero_mask.reshape(self.shape).unsqueeze(0).type(torch.int).to(self.device)
  
  def update_gmm(self):
    comp = D.Independent(
      D.Normal(self.means, self.covs), 1
    )
    gmm = D.MixtureSameFamily(self.mix, comp)
    return gmm

  def _pre_process(self, x):
    y = (x * 255. + torch.rand_like(x)) / 256.
    y = (2*y - 1) * self.data_constraint
    y = (y + 1) / 2
    y = y.log() - (1. - y).log()

    ldj = F.softplus(y) + F.softplus(-y) - F.softplus(
      (1. - self.data_constraint).log() - self.data_constraint.log()
    )
    sldj = ldj.view(ldj.size(0), -1).sum(-1)

    return y, sldj

  def forward(self, x):
    x, log_det_j = self._pre_process(x)
    # log_det_j = x.new_zeros(x.shape[0])
    for i, flow in enumerate(self.flows):
      x, det = flow.f(x, i%2 + (-1)**i * self.mask)
      log_det_j += det
    
    return x, log_det_j
  
  def reverse(self, z):
    i = len(self.flows) - 1
    for flow in reversed(self.flows):
      z = flow.g(z, i%2 + (-1)**i * self.mask)
      i -= 1
    return z
  
  def sample(self, sample_size):
    self.update_gmm()
    sample = self.prior.sample((sample_size,)).reshape((sample_size, *self.shape))
    i = len(self.flows) - 1
    for flow in reversed(self.flows):
      sample = flow.g(sample, i%2 + (-1)**i * self.mask)
      i -= 1
    return sample
  
  def log_prob(self, x):
    self.update_gmm()
    z, log_det_j = self.forward(x)
    # ll = -.5 * (z**2 + LOG_2_PI)
    # ll = ll.view(z.size(0), -1).sum(-1) - 256*np.prod(z.size()[1:])
    ll = self.prior.log_prob(z.view(z.size(0), -1))
    return ll + log_det_j
    
