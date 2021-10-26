import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.distributions as D
from acflow.coupling import AffineCoupling
from gmm import GaussianMixture

import numpy as np

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

    # self.means = nn.Parameter(torch.randn(n_comps, self.gmm_dim).to(device))
    # self.covs = nn.Parameter(torch.rand(n_comps, self.gmm_dim).to(device))
    
    # self.covs = torch.rand(n_comps, self.gmm_dim).to(device)
    # self.means = torch.eye(self.gmm_dim)[:n_comps].to(device)

    self.means = torch.randn(n_comps, self.gmm_dim).to(device)
    self.covs = torch.ones(n_comps, self.gmm_dim).to(device)

    self.mix = D.Categorical(torch.ones(self.n_comps,).to(self.device))
    self.update_gmm()


  @property
  def mask(self):
    temp = torch.arange(self.shape[0]*self.shape[1]*self.shape[2])
    zero_mask = temp%2 == 0
    return zero_mask.reshape(self.shape).unsqueeze(0).type(torch.int).to(self.device)
  
  def update_gmm(self):
    comp = D.Independent(
      D.Normal(self.means, self.covs), 1
    )
    self.prior = D.MixtureSameFamily(self.mix, comp)

  def forward(self, x):
    log_det_j = x.new_zeros(x.shape[0])
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
    sample = self.prior.sample(
      (sample_size,)
    ).reshape((sample_size, *self.shape))
    i = len(self.flows) - 1
    for flow in reversed(self.flows):
      sample = flow.g(sample, i%2 + (-1)**i * self.mask)
      i -= 1
    return sample
  
  def log_prob(self, x):
    self.update_gmm()
    z, log_det_j = self.forward(x)
    ll = self.prior.log_prob(z.view(z.size(0), -1))
    return ll + log_det_j - np.log(256) * np.prod(z.size()[1:])


class RealNVPFlows(nn.Module):

  def __init__(self, index, num_flows, in_channels, mid_channels, resnet_blocks):

    super(RealNVPFlows, self).__init__()

    self.is_last_flow = index == num_flows - 1

    self.in_couplings = nn.ModuleList(
      [
        AffineCoupling(in_channels, mid_channels, resnet_blocks),
        

      ]
    )


class OldRealNVP(nn.Module):

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

    self.prior = GaussianMixture(n_comps, np.prod(shape), 'diag', device=device)


  @property
  def mask(self):
    temp = torch.arange(self.shape[0]*self.shape[1]*self.shape[2])
    zero_mask = temp%2 == 0
    return zero_mask.reshape(self.shape).unsqueeze(0).type(torch.int).to(self.device)
  
  def update_gmm(self, x):
    self.prior.fit(x.view(x.size(0), -1))

  def forward(self, x):
    log_det_j = x.new_zeros(x.shape[0])
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
    sample = self.prior.sample(
      (sample_size,)
    ).reshape((sample_size, *self.shape))
    i = len(self.flows) - 1
    for flow in reversed(self.flows):
      sample = flow.g(sample, i%2 + (-1)**i * self.mask)
      i -= 1
    return sample
  
  def log_prob(self, x):
    z, log_det_j = self.forward(x)
    ll = self.prior._estimate_log_prob(z.view(z.size(0), -1))
    return ll + log_det_j - np.log(256) * np.prod(z.size()[1:])



