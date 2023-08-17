# Imports
import os
import torch
from loss import kld
from tqdm import tqdm
from model import ConvVAE
from utils import inference
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Model Parameters
imageSize = 64
channels = 3
h_dim = 1024
z_dim = 512
# Training
EPOCHS = 100
BATCH = 32
LR = 1e-3
TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(imageSize)])


# Datatset
dataset = datasets.LFWPeople(root = 'data/', split = 'train', transform = TRANSFORM, download = True)
trainLoader = DataLoader(dataset = dataset, batch_size = BATCH, shuffle = True)


# Training
def train(net, loader, opt, bce):
    loop = tqdm(enumerate(loader))
    for batch_idx, (x, _) in loop:
        x = x.to(device)

        # Forward Pass
        xRecons, mu, logvar = net(x)

        # Backwards
        loss = bce(xRecons, x) + kld(mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loop.set_postfix(lossVal = loss.item())


def main():
    model = ConvVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr = LR)
    loss = nn.BCELoss(reduction = 'sum')

    for epoch in range(EPOCHS):
        print(f'[{epoch+1}/{EPOCHS}] Epochs')
        train(model, trainLoader, optimizer, loss)
        if (epoch+1)%5 == 0:
            inference(model, epoch)

    return model

model = main()