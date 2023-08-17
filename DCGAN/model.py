import torch
from torch import nn


# this is correct
class Generator(nn.Module):
  def __init__(self, latentSize, ngf):
    super(Generator, self).__init__()
    self.main = nn.Sequential(
        nn.ConvTranspose2d(latentSize, ngf*8, 4, 1, 0, bias = False),
        nn.BatchNorm2d(ngf*8),
        nn.ReLU(True),

        # 512x4x4
        nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias = False),
        nn.BatchNorm2d(ngf*4),
        nn.ReLU(True),

        # 256x8x8
        nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias = False),
        nn.BatchNorm2d(ngf*2),
        nn.ReLU(True),

        # 128x16x16
        nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias = False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),

        # 64x32x32
        nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias = False),
        nn.Tanh()
        # 3x64x64
    )

  def forward(self, x):
    return self.main(x)


# this is correct
class Discriminator(nn.Module):
  def __init__(self, ndf):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
        nn.Conv2d(3, ndf, 4, 2, 1, bias = False),
        nn.LeakyReLU(0.2),

        # block 1
        nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias = False),
        nn.BatchNorm2d(ndf*2),
        nn.LeakyReLU(0.2, True),

        # block 2
        nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias = False),
        nn.BatchNorm2d(ndf*4),
        nn.LeakyReLU(0.2, True),

        # block 3
        nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias = False),
        nn.BatchNorm2d(ndf*8),
        nn.LeakyReLU(0.2, True),

        # block final
        nn.Conv2d(ndf*8, 1, 4, 1, 0, bias = False),
        nn.Sigmoid()
    )

  def forward(self, x):
    return self.main(x)
