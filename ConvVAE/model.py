import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, encoder = True, use_act = True, **kwargs):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(in_c, out_c, **kwargs) if encoder else nn.ConvTranspose2d(in_c, out_c, **kwargs)

    def forward(self, x):
        return F.relu(self.conv(x)) if self.use_act else self.conv(x)
    

class ConvVAE(nn.Module):
    def __init__(self, channels = 3, upChannels = 64, latent_dim = 100):
        super(ConvVAE, self).__init__()

        # Encoder
        self.encoderConv = nn.Sequential(
            ConvBlock(channels, upChannels, kernel_size = 4, stride = 2, padding = 2),
            ConvBlock(upChannels, upChannels*2, kernel_size = 4, stride = 2, padding = 2),
            ConvBlock(upChannels*2, upChannels*4, kernel_size = 4, stride = 2, padding = 2),
            ConvBlock(upChannels*4, upChannels*8, kernel_size = 4, stride = 2, padding = 2),
            ConvBlock(upChannels*8, 1024, kernel_size = 4, stride = 2, padding = 2)
        )
        self.conv2hid = nn.Linear(1024, 2048)
        self.hid2mu = nn.Linear(2048, latent_dim)
        self.hid2logvar = nn.Linear(2048, latent_dim)

        # Decoder
        self.z2hid = nn.Linear(latent_dim, 1024)
        self.decoderConv = nn.Sequential(
            ConvBlock(1024, upChannels*8, encoder = False, kernel_size = 3, stride = 2),
            ConvBlock(upChannels*8, upChannels*4, encoder = False, kernel_size = 3, stride = 2),
            ConvBlock(upChannels*4, upChannels*2, encoder = False, kernel_size = 3, stride = 2),
            ConvBlock(upChannels*2, upChannels, encoder = False, kernel_size = 3, stride = 2),
            ConvBlock(upChannels, channels, encoder = False, use_act = False, kernel_size = 4, stride = 2)
        )

    def encoder(self, x):
        x = self.encoderConv(x)
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        h = self.conv2hid(x)

        return self.hid2mu(h), self.hid2logvar(h)

    def decoder(self, z):
        h = self.z2hid(z)
        h = h.view(-1, 1024, 1, 1)

        return torch.sigmoid(self.decoderConv(h))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std

        return z

    def generate(self):
        std = torch.exp(0.5*self.logvar)
        eps = torch.randn_like(std)
        z = self.mu + eps*std

        return self.decoder(z)

    def forward(self, x):
        # Encoder
        self.mu, self.logvar = self.encoder(x)
        z = self.reparameterize(self.mu, self.logvar)
        # Decoder
        xRecons = self.decoder(z)

        return xRecons, self.mu, self.logvar


if __name__ == '__main__':
    x = torch.randn(16, 3, 64, 64)
    cvae = ConvVAE()
    x_recons, mu, sigma = cvae(x)
    print(x_recons.shape)
    print(mu.shape)
    print(sigma.shape)