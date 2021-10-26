import torch
import torch.nn as nn
import torch.nn.functional as F


class Rescale(nn.Module):

  def __init__(self, num_channels):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))
  
  def forward(self, x):
    x = self.weight * x
    return x

def checkerboard_mask(dim1, dim2, reverse, dtype=torch.float32, 
                      device=None, requires_grad=False):

  checkerboard = [[((i % 2) + j) % 2 for j in range(dim2)] for i in range(dim1)]

  if device is None:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

  mask = torch.Tensor(
    checkerboard, dtype=dtype, device=device, requires_grad=requires_grad
  )

  if reverse:
    mask = 1 - mask
  
  mask = mask.view(1,1,dim1,dim2)

  return mask

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
    