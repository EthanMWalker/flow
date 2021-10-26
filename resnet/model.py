import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet.block import ResNetBlock
from resnet.utils import WNConv2d

class ResNet(nn.Module):

  def __init__(self, in_channels, mid_channels, out_channels, num_blocks=3,
              kernel_size=1, padding=0, double=False):
    super().__init__()

    self.double = double

    self.blocks = nn.ModuleList(
      [
        ResNetBlock(mid_channels, mid_channels) for _ in range(num_blocks)
      ]
    )

    self.skips = nn.ModuleList(
      [
        WNConv2d(
          mid_channels, mid_channels, kernel_size=1, padding=0, bias=True
        ) for _ in range(num_blocks)
      ]
    )

    self.in_norm = nn.BatchNorm2d(in_channels)
    self.in_layer = WNConv2d(
      2*in_channels, mid_channels, kernel_size, padding=padding, bias=True
    )
    self.in_skip = WNConv2d(
      mid_channels, mid_channels, kernel_size=1, padding=0, bias=True
    )

    self.out_norm = nn.BatchNorm2d(mid_channels)
    self.out_layer = WNConv2d(
      mid_channels, out_channels, kernel_size=1, padding=0
    )
  

  def forward(self, x):
    x = self.in_norm(x)
    if self.double:
      x *= 2.
    x = torch.cat((x,-x), dim=1)
    x = F.relu(x)
    x = self.in_layer(x)
    x_skip = self.in_skip(x)

    for block, skip in zip(self.blocks, self.skips):
      x = block(x)
      x_skip += skip(x)

    x = self.out_norm(x_skip)
    x = F.relu(x)
    x = self.out_layer(x)

    return x


