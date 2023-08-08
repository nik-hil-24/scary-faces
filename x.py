import os
import torch
import numpy as np
from tqdm import tqdm
from model import CVAE
from torch import nn, optim
from matplotlib import animation
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms, datasets

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Model Parameters
image_size = 64
channels = 3
h_dim = 256
z_dim = 64
# Training
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3           # Karpathy Constant
TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(image_size),
                                transforms.Normalize(0.5, 0.5, 0.5)])
class_dim = 2

# Datatset
dataset = datasets.CelebA('data', transform = TRANSFORM, download=True)
train_loader = DataLoader(dataset = dataset, batch_size = 16, shuffle=True)

_, classes = next(iter(train_loader))

# OHE Labels
def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets

# Training
def train(net, loader, opt, loss_fn):
    loop = tqdm(enumerate(loader))
    for batch_idx, (x, y) in loop:
        # Forward Pass
        x, y = x.to(device), one_hot(y[:, 20], class_dim).to(device)
        x_recons, mu, sigma = net(x, y)

        # Loss
        recons_loss = loss_fn(x_recons, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        # Backward Pass
        loss = recons_loss + kl_div
        opt.zero_grad()
        loss.backward()
        opt.step()
        loop.set_postfix(loss = loss.item())

def main():
    model = CVAE(channels, image_size, class_dim, h_dim, z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr = LR)
    loss = nn.BCELoss(reduction = 'sum')
    
    for epoch in range(EPOCHS):
        print(f'[{epoch+1}/{EPOCHS}] Epochs')
        train(model, train_loader, optimizer, loss)

    return model

model = main()


# Saving Generated Images
def inference():
    # Folder for Images to be Stored in
    if 'inference' not in os.listdir():
        os.mkdir('inference')
    
    dim = model.mu.shape[0]
    
    # Each Class Label
    for c in range(class_dim):
        label = one_hot(torch.tensor([c for _ in range(dim)]), class_dim)
        # Generate Image
        images = model.generate(label)
        # Save Images
        for i in range(2):
            out = images[i].view(-1, 1, 28, 28)
            save_image(out, f"inference/generated_{c}_iter_{i+1}.png")
    
inference()