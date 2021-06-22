import torch
import torch.nn as nn
from resnet import ResNet
from acflow.utils import MLP, Rescale

class AffineCoupling(nn.Module):

  def __init__(self, in_channels, mid_channels, num_blocks=4):
    super().__init__()

    self.s = ResNet(in_channels, mid_channels, in_channels, num_blocks)
    self.t = ResNet(in_channels, mid_channels, in_channels, num_blocks)
    self.rescale = nn.utils.weight_norm(Rescale(in_channels))
  
  def f(self, x, mask):
    z = x
    z_ = x*mask
    s = self.s(z_)*(1-mask)
    s = self.rescale(torch.tanh(s))
    t = self.t(z_)*(1-mask)
    z = z * s.exp() + t
    return z, s.view(s.size(0), -1).sum(-1)

  def g(self, z, mask):
    x = z
    x_ = z*mask
    s = self.s(x_) * (1-mask)
    s = self.rescale(torch.tanh(s))
    t = self.t(x_) * (1-mask)
    x = (x - t) * s.mul(-1).exp()
    return x

class AffineCouplingLinear(nn.Module):

  def __init__(self, in_dim, mid_dim, num_layers=3):
    super().__init__()

    self.s = MLP(in_dim, mid_dim, in_dim, num_layers)
    self.t = MLP(in_dim, mid_dim, in_dim, num_layers)
  
  def f(self, x, mask):
    z = x
    z_ = x*mask
    s = self.s(z_)*(1-mask)
    t = self.t(z_)*(1-mask)
    z = z * s.exp() + t
    return z, s.view(s.size(0), -1).sum(-1)

  def g(self, z, mask):
    x = z
    x_ = z*mask
    s = self.s(x_) * (1-mask)
    t = self.t(x_) * (1-mask)
    x = (x - t) * s.mul(-1).exp()
    return x