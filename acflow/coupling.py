import torch
import torch.nn as nn
from resnet import ResNet
from acflow.utils import Rescale, checkerboard_mask, MaskType

class AffineCoupling(nn.Module):

  def __init__(self, in_channels, mid_channels, mask_type, reverse_mask,
                num_blocks=6):
    super().__init__()

    self.st = ResNet(
      in_channels, mid_channels, 2 * in_channels, num_blocks, 3, 1,
      double=(mask_type == MaskType.CHECKERBOARD)
    )
    self.rescale = nn.utils.weight_norm(Rescale(in_channels))
    self.mask_type = mask_type
    self.reverse_mask = reverse_mask
  
  def forward(self, x, log_det=None, reverse=True):
    if self.mask_type == MaskType.CHECKERBOARD:
      mask = checkerboard_mask(
        x.size(2), x.size(3), self.reverse_mask, device=x.device
      )

      x_ = x*mask
      st = self.st(x_)
      s, t = st.chunk(2, dim=1)
      s = self.rescale(torch.tanh(s))
      s = s*(1-mask)
      t = t*(1-mask)

      if reverse:
        x = x * s.mul(-1).exp() - t
      else:
        x = (x + t) * s.exp()
        log_det += s.view(s.size(0), -1).sum(-1)

    
    else:
      if self.reverse_mask:
        x_id, x_change = x.chunk(2, dim=1)
      else:
        x_change, x_id = x.chunk(2, dim=1)
      
      st = self.st(x_id)
      s, t = st.chunk(2, dim=1)
      s = self.rescale(torch.tanh(s))
      x_change = (x_change + t) * s.exp()

      if reverse:
        x_change = x_change * s.mul(-1).exp() - t
      else:
        x_change = (x_change + t) * s.exp()
        log_det += s.view(s.size(0), -1).sum(-1)

      if self.reverse_mask:
        x = torch.cat((x_id, x_change), dim=1)
      else:
        x = torch.cat((x_change, x_id), dim=1)
        
    return x, log_det



