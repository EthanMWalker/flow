import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import IntEnum

class MaskType(IntEnum):
  CHECKERBOARD = 0
  CHANNEL_WISE = 1

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

  mask = torch.tensor(
    checkerboard, dtype=dtype, device=device, requires_grad=requires_grad
  )

  if reverse:
    mask = 1 - mask
  
  mask = mask.view(1,1,dim1,dim2)

  return mask

def squeeze2x2(x, reverse=False, alt_order=False):
  block_size = 2
  alt_order = False
  if alt_order:
    n, c, h, w = x.size()

    if reverse:
      if c %4 != 0:
        raise ValueError('bad c val')
      c //= 4
    else:
      if h % 2 != 0:
        raise ValueError('bad h val')

      if w % 2 != 0:
        raise ValueError('bad w val')
    thing = [
        [[[1., 0.], [0., 0.]]],
        [[[0., 0.], [0., 1.]]],
        [[[0., 1.], [0., 0.]]],
        [[[0., 0.], [1., 0.]]]
    ]
    squeeze_matrix = torch.tensor(thing, dtype=x.dtype, device=x.device)
    perm_weight = torch.zeros((4*c, c, 2, 2), dtype=x.dtype, device=x.device)

    for i in range(c):
      slice_0 = slice(i*4, (i+1)*4)
      slice_1 = slice(i, i+1)
      perm_weight[slice_0, slice_1, :, :] = squeeze_matrix
    shuffle_channels = torch.tensor(
      [i*4 for i in range(c)]
      + [i*4 + 1 for i in range(c)]
      + [i*4 + 2 for i in range(c)]
      + [i*4 + 3 for i in range(c)]
    )
    perm_weight = perm_weight[shuffle_channels, :, :, :]

    if reverse:
      x = F.conv_transpose2d(x, perm_weight, stride=2)
    else:
      x = F.conv2d(x, perm_weight, stride=2)
      
  else:
    b, c, h, w = x.size()
    x = x.permute(0,2,3,1)

    if reverse:
      if c %4 != 0:
        raise ValueError('Bad number of channels, not divisible by 4')
      x = x.view(b, h, w, c//4, 2, 2)
      x = x.permute(0,1,4,2,5,3)
      x = x.contiguous().view(b, 2*h, 2*w, c//4)

    else:
      if h%2 != 0 or w%2 != 0:
        raise ValueError('not even spatial dims')
      x = x.view(b, h//2, 2, w//2, 2, c)
      x = x.permute(0,1,3,5,2,4)
      x = x.contiguous().view(b, h//2, w//2, c*4)
    
    x = x.permute(0,3,1,2)

  return x

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
