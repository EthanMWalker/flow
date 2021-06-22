import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

  def __init__(self, in_dim, mid_dim, out_dim, num_layers=3):
    super().__init__()

    self.in_layer = nn.Linear(in_dim, mid_dim)

    self.mid_layers = nn.ModuleList(
      [
        nn.Linear(mid_dim, mid_dim) for _ in range(num_layers)
      ]
    )

    self.out_layer = nn.Linear(mid_dim, out_dim)

  
  def forward(self, x):
    x = self.in_layer(x)
    x = F.relu(x)

    for layer in self.mid_layers:
      x = layer(x)
      x = F.relu(x)
    
    x = self.out_layer(x)
    x = F.relu(x)
    return x

class Rescale(nn.Module):

  def __init__(self, num_channels):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))
  
  def forward(self, x):
    x = self.weight * x
    return x