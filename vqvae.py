import torch
import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self, in_ch, out_ch, k = 3, s = 1, p = 1, use_gn = True, gn_groups = 8):
    super().__init__()

    self.conv = nn.Conv2d(in_ch, out_ch, k, stride = s, padding = p)

    self.norm = None
    if use_gn:
      g = min(gn_groups, out_ch)
      while out_ch % g != 0 and g > 1:
        g -= 1
      self.norm = nn.GroupNorm(g, out_ch)
    else:
      self.norm = nn.BatchNorm2d(out_ch)

    self.act = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.conv(x)
    if self.norm is not None:
      x = self.norm(x)
    return self.act(x)

class Encoder(nn.Module):
  def __init__(self, in_channels=1, hidden=128, z_channels = 64):
    super().__init__()

    self.net = nn.Sequential(
        ConvBlock(in_channels, hidden // 2, k = 4, s = 2, p = 1),
        ConvBlock(hidden // 2, hidden, k = 4, s = 2, p = 1),
        ConvBlock(hidden, hidden, k = 4, s = 2, p = 1),
        ConvBlock(hidden, hidden, k = 3, s = 1, p = 1),
        nn.Conv2d(hidden, z_channels, kernel_size=1)
    )
  def forward(self, x):
    return self.net(x)

class Decoder(nn.Module):
  def __init__(self, out_channels = 1, hidden = 128, z_channels = 64):
    super().__init__()

    self.net = nn.Sequential(
        ConvBlock(z_channels, hidden, k = 3, s = 1, p = 1),
        ConvBlock(hidden, hidden, k = 3, s = 1, p = 1),

        nn.ConvTranspose2d(hidden, hidden, kernel_size=4, stride = 2, padding=1),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(hidden, hidden, kernel_size=4, stride = 2, padding=1),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(hidden, hidden//2, kernel_size=4, stride = 2, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(hidden // 2, out_channels, 1)
    )
  def forward(self, x):
    return self.net(x)

class VQVAE(nn.Module):
  def __init__(self, quantizer, in_channels = 1, hidden = 128, z_channels = 64, num_groups=8):
    super().__init__()

    self.encoder = Encoder(in_channels=in_channels, hidden=hidden, z_channels=z_channels)
    self.pre_vq_norm = nn.GroupNorm(num_groups=num_groups, num_channels=z_channels)

    self.quantizer = quantizer

    self.decoder = Decoder(out_channels=in_channels, hidden=hidden, z_channels=z_channels)

  def forward(self, x):
    z_e = self.encoder(x)
    z_e = self.pre_vq_norm(z_e)
    z_q, vq_loss, perplexity, indices = self.quantizer(z_e)
    x_recon = self.decoder(z_q)

    return x_recon, vq_loss, perplexity, indices

