import torch
from torch import nn


# x -> hidden dim -> mean, std -> Parametrization trick -> decoder -> output image
class CVAE(nn.Module):
    def __init__(self, in_channels, image_size, class_dim, hidden_dim = 200, z_dim = 20):
        super().__init__()
        # Sizes
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_dim = None
        self.class_dim = class_dim
        # Initialize
        self.relu = nn.ReLU()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 32 x 32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 16 x 16
        )
        if self.feature_dim == None:
            self.feature_dim = self.get_flat_shape()
        self.image_2hid = nn.Linear(self.feature_dim+class_dim, hidden_dim)
        self.hid_2mu = nn.Linear(hidden_dim, z_dim)
        self.hid_2sigma = nn.Linear(hidden_dim, z_dim)
        
        
        # Decoder
        self.z_2hid = nn.Linear(z_dim+class_dim, hidden_dim)
        self.hid_2img = nn.Linear(hidden_dim, self.feature_dim)
        self.decoder_conv = nn.Sequential(
            # in: latent_size x 1 x 1
            nn.ConvTranspose2d(self.feature_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # output is between -1 to 1
            # out: 3 x 64 x 64
        )
        
    def get_flat_shape(self):
        x = torch.randn(1, 3, 64, 64)
        _, c, h, w = self.encoder_conv(x).shape
        
        return c*h*w
    
    @staticmethod
    def flatten(x):
        batch_size, c, h, w = x.shape
        
        return x.view(batch_size, h*w*c)
    
    def encoder(self, x, c):
        # q_phi(z/x, c)
        x = self.encoder_conv(x)
        x = self.flatten(x)

        inputs = torch.cat([x, c], 1)
        h = self.relu(self.image_2hid(inputs))
        self.mu = self.hid_2mu(h)
        self.sigma = self.hid_2sigma(h)
        
    def decoder(self, z, c):
        # p_theta(x/z, c)
        inputs = torch.cat([z, c], 1)
        h = self.relu(self.z_2hid(inputs))
        h = torch.sigmoid((self.hid_2img(h)))
        # Add dims
        h = h.unsqueeze(2).unsqueeze(3)
        op = self.decoder_conv(h)

        return op
        
    def reparametrize(self):
        # Reparametrization for Gaussian Distribution
        eps = torch.rand_like(self.sigma)
        
        return self.mu + self.sigma*eps
    
    def generate(self, c):
        # Generate Data
        z = self.reparametrize()
        
        return self.decoder(z, c)
         
    def forward(self, x, c):
        # Encode
        self.encoder(x, c)
        # Reparametrize
        z = self.reparametrize()
        # Decode
        x_recons = self.decoder(z, c)
        
        return x_recons, self.mu, self.sigma
        
        
if __name__ == '__main__':
    x = torch.randn(2, 3, 64, 64)
    c = torch.tensor([[0,1], [1,0]])
    cvae = CVAE(3, 64, 2)
    x_recons, mu, sigma = cvae(x, c)
    print(x_recons.shape)
    print(mu.shape)
    print(sigma.shape)