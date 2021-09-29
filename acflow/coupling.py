import torch
import torch.nn as nn
from resnet import ResNet
from acflow.utils import MLP, Rescale

class AffineCoupling(nn.Module):

  def __init__(self, in_channels, mid_channels, num_blocks=6):
    super().__init__()

    self.st = ResNet(in_channels, mid_channels, 2 * in_channels, num_blocks)
    self.rescale = nn.utils.weight_norm(Rescale(in_channels))
  
  def f(self, x, mask):
    z = x
    z_ = x*mask
    st = self.st(z_)
    s, t = st.chunk(2, dim=1)
    s = self.rescale(torch.tanh(s))
    s = s*(1-mask)
    t = t*(1-mask)
    z = (z + t) * s.exp()
    return z, s.view(s.size(0), -1).sum(-1)

  def g(self, z, mask):
    x = z
    x_ = z*mask
    st = self.st(x_)
    s, t = st.chunk(2, dim=1)
    s = self.rescale(torch.tanh(s))
    s = s*(1-mask)
    t = t*(1-mask)
    x = x * s.mul(-1).exp() - t
    return x

