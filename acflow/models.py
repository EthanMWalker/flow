import torch
import torch.nn as nn
from acflow.coupling import AffineCoupling


class RealNVP(nn.Module):

  def __init__(self, in_channels, mid_channels, num_layers, prior, shape, device):
    super().__init__()

    self.prior = prior

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
    i = 0
    log_det_j = x.new_zeros(x.shape[0])
    for flow in reversed(self.flows):
      x, det = flow.f(x, i%2 + (-1)**i * self.mask)
      log_det_j += det
      i += 1
    
    return x, log_det_j
  
  def sample(self, sample_size):
    sample = self.prior.sample((sample_size,)).reshape((sample_size, *self.shape))
    for i, flow in enumerate(self.flows):
      sample = flow.g(sample, i%2 + (-1)**i * self.mask)
    return sample
  
  def log_prob(self, x):
    z, log_det_j = self.forward(x)
    return self.prior.log_prob(torch.flatten(z,1)) + log_det_j
    