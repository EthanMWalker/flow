import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from resnet.utils import WNConv2d

class ResNetBlock(nn.Module):

  def __init__(self, in_channels, out_channels):

    super(ResNetBlock, self).__init__()

    self.norms = nn.ModuleList(
      [
        nn.BatchNorm2d(in_channels),
        nn.BatchNorm2d(out_channels)
      ]
    )

    self.layers = nn.ModuleList(
      [
        WNConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        WNConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
      ]
    )
  
  def forward(self, x):
    residual = x

    for layer, norm in zip(self.layers, self.norms):
      x = norm(x)
      x = F.relu(x)
      x = layer(x)
    
    x = x + residual
    return x
