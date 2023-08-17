import os
import torch
from torch import nn
from torchvision.utils import save_image

def weightInit(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    
def inference(gen, epoch, device, n = 2):
    if 'gan' not in os.listdir():
        try:
            os.mkdir('gan')
        except:
            pass

    for i in range(n):
        x = torch.randn(1, 100, 1, 1).to(device)
        y = gen(x)
        y = y.cpu()
        save_image(y.view(3,64,64), f"gan/generated_epoch_{epoch}_iter_{i+1}.png")