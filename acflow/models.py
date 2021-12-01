import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.distributions as D
from acflow.coupling import AffineCoupling
from acflow.utils import MaskType, squeeze2x2
from gmm import GaussianMixture

import numpy as np

class RealNVP(nn.Module):
  def __init__(self, in_channels, mid_channels, num_layers, n_comps, 
              shape, device, num_blocks=8):
    super().__init__()

    self.num_layers = num_layers
    self.n_comps = n_comps
    self.gmm_dim = np.prod(shape)
    self.shape = shape
    self.device = device

    self.flows = RealNVPFlows(
      0, num_layers, in_channels, mid_channels, num_blocks
    )

    # self.means = nn.Parameter(torch.randn(n_comps, self.gmm_dim).to(device))
    # self.covs = nn.Parameter(torch.rand(n_comps, self.gmm_dim).to(device))
    
    # self.means = torch.eye(self.gmm_dim)[:n_comps].to(device)
    # self.covs = torch.rand(n_comps, self.gmm_dim).to(device)

    self.means = torch.randn(n_comps, self.gmm_dim).to(device)
    self.covs = torch.ones(n_comps, self.gmm_dim).to(device)

    self.mix = D.Categorical(torch.ones(self.n_comps,).to(self.device))
    self.update_gmm()
  
  def update_gmm(self):
    comp = D.Independent(
      D.Normal(self.means, self.covs), 1
    )
    self.prior = D.MixtureSameFamily(self.mix, comp)
  
  def forward(self, x, reverse=False):
    log_det = None

    if not reverse:
      log_det = 0.
    
    x, log_det = self.flows(x, log_det, reverse)

    return x, log_det
  
  def sample(self, sample_size):
    self.update_gmm()

    # sample = self.prior.sample(
    #   (sample_size,)
    # ).reshape((sample_size, *self.shape))

    sample = torch.randn((sample_size, *self.shape), device=self.device)

    sample, log_det = self.forward(sample, reverse=True)
    return sample
  
  # def log_prob(self, x):
  #   self.update_gmm()
  #   z, log_det_j = self.forward(x, reverse=False)
  #   ll = self.prior.log_prob(z.view(z.size(0), -1))
  #   ll = ll + log_det_j - np.log(256) * np.prod(z.size()[1:])
  #   return -ll.mean()

  def log_prob(self, x):
    z, log_det = self.forward(x, reverse=False)
    prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
    prior_ll = prior_ll.view(z.size(0), -1).sum(-1) \
        - np.log(256) * np.prod(z.size()[1:])
    ll = prior_ll + log_det
    nll = -ll.mean()
    return nll


class RealNVPFlows(nn.Module):

  def __init__(self, index, num_flows, in_channels, mid_channels, resnet_blocks):

    super(RealNVPFlows, self).__init__()

    self.is_last_flow = index == num_flows - 1

    self.in_couplings = nn.ModuleList(
      [
        AffineCoupling(
          in_channels, mid_channels, MaskType.CHECKERBOARD, reverse_mask=False,
          num_blocks=resnet_blocks
        ),
        AffineCoupling(
          in_channels, mid_channels, MaskType.CHECKERBOARD, reverse_mask=True,
          num_blocks=resnet_blocks
        ),
        AffineCoupling(
          in_channels, mid_channels, MaskType.CHECKERBOARD, reverse_mask=False,
          num_blocks=resnet_blocks
        )
      ]
    )

    if self.is_last_flow:
      self.in_couplings.append(
        AffineCoupling(
          in_channels, mid_channels, MaskType.CHECKERBOARD, reverse_mask=True, 
          num_blocks=resnet_blocks
        )
      )
    else:
      self.out_couplings = nn.ModuleList(
        [
          AffineCoupling(
            4*in_channels, 2*mid_channels, MaskType.CHANNEL_WISE, 
            reverse_mask=False, num_blocks=resnet_blocks
          ),
          AffineCoupling(
            4*in_channels, 2*mid_channels, MaskType.CHANNEL_WISE, 
            reverse_mask=False, num_blocks=resnet_blocks
          ),
          AffineCoupling(
            4*in_channels, 2*mid_channels, MaskType.CHANNEL_WISE, 
            reverse_mask=False, num_blocks=resnet_blocks
          )
        ]
      )
      self.next_flow = RealNVPFlows(
        index+1, num_flows, 2*in_channels, 2*mid_channels, resnet_blocks
      )
  
  def forward(self, x, log_det, reverse=False):
    if reverse:
      if not self.is_last_flow:
        x = squeeze2x2(x, reverse=False, alt_order=True)
        x, x_split = x.chunk(2, dim=1)
        x, log_det = self.next_flow(x, log_det, reverse)
        x = torch.cat((x, x_split), dim=1)
        x = squeeze2x2(x, reverse=True, alt_order=True)

        x = squeeze2x2(x, reverse=False)
        for coupling in reversed(self.out_couplings):
          x, log_det = coupling(x, log_det, reverse)
        x = squeeze2x2(x, reverse=True)

      for coupling in reversed(self.in_couplings):
        x, log_det = coupling(x, log_det, reverse)
    
    else:
      for coupling in self.in_couplings:
        x, log_det = coupling(x, log_det, reverse)
      
      if not self.is_last_flow:
        x = squeeze2x2(x, reverse=False)
        for coupling in self.out_couplings:
          x, log_det = coupling(x, log_det, reverse)
        x = squeeze2x2(x, reverse=True)

        x = squeeze2x2(x, reverse=False, alt_order=True)
        x, x_split= x.chunk(2, dim=1)
        x, log_det = self.next_flow(x, log_det, reverse)
        x = torch.cat((x, x_split), dim=1)
        x = squeeze2x2(x, reverse=True, alt_order=True)
    
    return x, log_det

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



