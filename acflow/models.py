import torch
import torch.nn as nn
from acflow.coupling import AffineCoupling, AffineCouplingLinear

import numpy as np

LOG_2_PI = np.log(2*np.pi)


class RealNVP(nn.Module):

  def __init__(self, in_channels, mid_channels, num_layers, prior, shape, device):
    super().__init__()

    self.prior = prior
    self.num_layers = num_layers

    self.flows = nn.ModuleList(
      [
        AffineCoupling(in_channels, mid_channels) for _ in range(num_layers)
      ]
    )
    self.shape = shape
    self.device = device
  
  @property
  def mask(self):
    temp = torch.arange(self.shape[0]*self.shape[1]*self.shape[2])
    zero_mask = temp%2 == 0
    return zero_mask.reshape(self.shape).unsqueeze(0).type(torch.int).to(self.device)


  def forward(self, x):
    log_det_j = x.new_zeros(x.shape[0])
    for i, flow in enumerate(self.flows):
      x, det = flow.f(x, i%2 + (-1)**i * self.mask)
      log_det_j += det
    
    return x, log_det_j
  
  def sample(self, sample_size):
    sample = self.prior.sample((sample_size,)).reshape((sample_size, *self.shape))
    i = len(self.flows) - 1
    for flow in reversed(self.flows):
      sample = flow.g(sample, i%2 + (-1)**i * self.mask)
      i -= 1
    return sample
  
  def log_prob(self, x):
    z, log_det_j = self.forward(x)
    ll = -.5 * (z**2 + LOG_2_PI)
    ll = ll.view(z.size(0), -1).sum(-1) - 256*np.prod(z.size()[1:])
    return ll + log_det_j
    # return self.prior.log_prob(torch.flatten(z,1)) + log_det_j
    

class RealNVPLinear(nn.Module):

  def __init__(self, in_channels, mid_channels, num_layers, prior, shape, device):
    super().__init__()

    self.prior = prior

    self.flows = nn.ModuleList(
      [
        AffineCouplingLinear(in_channels, mid_channels) for _ in range(num_layers)
      ]
    )
    self.shape = shape
    self.device = device
  
  @property
  def mask(self):
    temp = torch.arange(self.shape[0]*self.shape[1])
    zero_mask = temp%2 == 0
    return zero_mask.unsqueeze(0).type(torch.int).to(self.device)


  def forward(self, x):
    i = 0
    log_det_j = x.new_zeros(x.shape[0])
    for flow in reversed(self.flows):
      x, det = flow.f(x, i%2 + (-1)**i * self.mask)
      log_det_j += det
      i += 1
    
    return x, log_det_j
  
  def sample(self, sample_size):
    sample = self.prior.sample((sample_size,))
    for i, flow in enumerate(self.flows):
      sample = flow.g(sample, i%2 + (-1)**i * self.mask)
    return sample
  
  def log_prob(self, x):
    z, log_det_j = self.forward(x)
    ll = -.5 * (z**2 + LOG_2_PI)
    ll = ll.sum()
    return ll + log_det_j