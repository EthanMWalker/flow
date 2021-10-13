import torch
import torch.nn as nn


class WNConv2d(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, padding, 
                bias=True):
    super(WNConv2d, self).__init__()

    self.conv = nn.utils.weight_norm(
      nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=padding, bias=bias
      )
    )

  def forward(self, x):
    return self.conv(x)