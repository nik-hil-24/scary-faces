import torch
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import weightInit, inference
from model import Generator, Discriminator
from torchvision import transforms, datasets


# Parameters

# device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  

# Training batch size
batchSize = 128

# Latent vector size
latentSize = 100

# Number of channels
nc = 3

# Image Size
imageSize = 64

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Learning Rate and hyperparameters
lr = 0.0002
beta1 = 0.5

# Epochs
epochs = 100

# Transforms
TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(imageSize)])


# Datatset
dataset = datasets.LFWPeople(root = 'data', split = 'train', transform = TRANSFORM, download = False)
trainLoader = DataLoader(dataset = dataset, batch_size = batchSize, shuffle = True)


# Model

generator = Generator(latentSize, ngf).to(device)
generator.apply(weightInit)

discriminator = Discriminator(ndf).to(device)
discriminator.apply(weightInit)


# Train

criterion = nn.BCELoss()

noise = torch.randn(32, latentSize, 1, 1)

optimizerD = optim.Adam(discriminator.parameters(), lr = lr, betas = (beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr = lr,  betas = (beta1, 0.999))

errorD = []
errorG = []

discriminator.train()
generator.train()

for epoch in range(epochs):
    print(f'[{epoch+1}/{epochs}] Epochs')
    loop = tqdm(enumerate(trainLoader))
    # for i in range(0, len(data), batchSize):
    for batch_idx, (x, _) in loop:
        X = x.to(device)
        b, _, _, _ = X.shape
        noise = torch.randn(b, latentSize, 1, 1).to(device)
        realLabels = torch.ones(b).to(device)
        fakeLabels = torch.zeros(b).to(device)

        # Train Discriminator
        discriminator.zero_grad()
        fake = generator(noise)
        output = discriminator(X).view(-1).to(device)
        errR = criterion(output, realLabels)

        fake = generator(noise)
        output = discriminator(fake).view(-1).to(device)
        errF = criterion(output, fakeLabels)

        errD = errR + errF
        errD.backward(retain_graph = True)
        optimizerD.step()

        # Train Generator
        generator.zero_grad()
        noise = torch.randn(b, latentSize, 1, 1).to(device)
        fake = generator(noise)
        realLabels = torch.ones(b).to(device)
        output = discriminator(fake).view(-1).to(device)
        errG = criterion(output, realLabels)

        errG.backward()
        optimizerG.step()

        errorD.append(errD.item())
        errorG.append(errG.item())
        loop.set_postfix(lossD = errD.item(), lossG = errG.item())

    if (epoch+1)%10 == 0:
        inference(generator, epoch, device)

inference(generator, 'trained', device, 10)

plt.plot(errorD)
plt.plot(errorG, alpha = 0.7)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Discriminator', 'Generator'])
plt.show()