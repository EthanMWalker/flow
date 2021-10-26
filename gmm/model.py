import torch
import torch.nn as nn
import numpy as np


class GaussianMixture(nn.Module):

  def __init__(self, n_comps, n_dims, cov_type='full', mu=None, var=None):
    super(GaussianMixture, self).__init__()


    self.n_comps = n_comps
    self.n_dims = n_dims
    self.cov_type = cov_type

    self.mu = mu
    self.var = var

    self._init_params()
  

  def _init_params(self):
    if self.mu is None:
      self.mu = nn.Parameter(
        torch.randn(1, self.n_comps, self.n_dims), requires_grad=False
      )
    else:
      self.mu = nn.Parameter(self.mu, requires_grad=False)
    
    if self.cov_type == 'full':
      if self.var is None:
        self.var = nn.Parameter(
          torch.eye(
            self.n_dims, dtype=torch.float64
          ).reshape(1,1,self.n_dims,self.n_dims).repeat(1,self.n_components,1,1),
          requires_grad=False
        )
      else:
        self.var = nn.Parameter(self.var, requires_grad=False)
    
    elif self.cov_type == 'diag':
      if self.var is None:
        self.var = nn.Parameter(
          torch.ones(1, self.n_comps, self.n_dims), equires_grad=False
        )
      else:
        self.var = nn.Parameter(self.var, requires_grad=False)
      
    self.pi = nn.Parameter(
      torch.Tensor(1, self.n_comps, 1), requires_grad=False
    ).fill_(1. / self.n_comps)