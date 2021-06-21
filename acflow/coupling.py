import torch
import torch.nn as nn
from resnet import ResNet

class AffineCoupling(nn.Module):

  def __init__(self, in_channels, mid_channels):
    super().__init__()

    self.s = ResNet(in_channels, mid_channels, in_channels)
    self.t = ResNet(in_channels, mid_channels, in_channels)
  
  def g(self, z, mask):
    x = z
    x_ = x*mask
    s = self.s(x_)*(1 - mask)
    t = self.t(x_)*(1 - mask)
    x = x_ + (1 - mask) * (x + torch.exp(s) + t)
    return x

  def f(self, x, mask):
    z =x
    z_ = mask * z
    s = self.s(z_) * (1-mask)
    t = self.t(z_) * (1-mask)
    z = (1 - mask) * (z - t) * torch.exp(-s) + z_
    return z, -s.sum(dim=(1,2,3))