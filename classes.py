import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob


######################################################################################
# Data Classes
######################################################################################
class FruitTrainDataset(Dataset):
    def __init__(self, files, shuffle, split_val, class_names, transform=transforms.ToTensor()):
        self.shuffle = shuffle
        self.class_names = class_names
        self.split_val = split_val
        self.data = np.array([files[i] for i in shuffle[split_val:]])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        name = self.data[idx].split('/')[-2]
        y = self.class_names.index(name)
        img = self.transform(img)

        return img, y


class FruitValidDataset(Dataset):
    def __init__(self, files, shuffle, split_val, class_names, transform=transforms.ToTensor()):
        self.shuffle = shuffle
        self.class_names = class_names
        self.split_val = split_val
        self.data = np.array([files[i] for i in shuffle[:split_val]])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        name = self.data[idx].split('/')[-2]
        y = self.class_names.index(name)
        img = self.transform(img)

        return img, y


class FruitTestDataset(Dataset):
    def __init__(self, path, class_names, transform=transforms.ToTensor()):
        self.class_names = class_names
        self.data = np.array(glob(os.path.join(path, '*/*.jpg')))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        name = self.data[idx].split('/')[-2]
        y = self.class_names.index(name)
        img = self.transform(img)

        return img, y


######################################################################################
# Network classes
######################################################################################
class Generator(nn.Module):
    def __init__(self, inp=1000):
        super(Generator, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(inp, inp//2, 5, 1, 0)
        self.conv1 = nn.Conv2d(inp//2, inp//2, 3, 1, 1)
        self.batchn1 = nn.BatchNorm2d(inp//2)

        self.tconv2 = nn.ConvTranspose2d(inp//2, inp//4, 4, 2, 0)
        self.conv2 = nn.Conv2d(inp//4, inp//4, 3, 1, 1)
        self.batchn2 = nn.BatchNorm2d(inp//4)

        self.tconv3 = nn.ConvTranspose2d(inp//4, inp//8, 4, 2, 0)
        self.conv3 = nn.Conv2d(inp//8, inp//8, 3, 1, 1)
        self.batchn3 = nn.BatchNorm2d(inp//8)

        self.convc = nn.Conv2d(inp//8, inp//16, 2, 1, 0)

        self.tconv4 = nn.ConvTranspose2d(inp//16, inp//32, 4, 2, 1)
        self.conv4 = nn.Conv2d(inp//32, inp//32, 3, 1, 1)
        self.batchn4 = nn.BatchNorm2d(inp//32)

        self.tconv5 = nn.ConvTranspose2d(inp//32, 3, 4, 2, 1)
        self.conv5 = nn.Conv2d(3, 3, 3, 1, 1)
        self.batchn5 = nn.BatchNorm2d(3)

        # functions
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def decode(self, z):
        z = self.relu(self.batchn1(self.conv1(self.tconv1(z))))
        z = self.relu(self.batchn2(self.conv2(self.tconv2(z))))
        z = self.relu(self.batchn3(self.conv3(self.tconv3(z))))
        z = self.tanh(self.convc(z))
        z = self.relu(self.batchn4(self.conv4(self.tconv4(z))))
        z = self.relu(self.batchn5(self.conv5(self.tconv5(z))))
        return z

    def forward(self, z):
        return self.decode(z)


class Discriminator(nn.Module):
    def __init__(self, inc=8):
        super(Discriminator, self).__init__()

        self.sconv1 = nn.Conv2d(3, 2*inc, 4, 2, 1)
        self.conv1 = nn.Conv2d(2*inc, 2*inc, 3, 1, 1)
        self.batchn1 = nn.BatchNorm2d(2*inc)

        self.sconv2 = nn.Conv2d(2*inc, 4*inc, 4, 2, 1)
        self.conv2 = nn.Conv2d(4*inc, 4*inc, 3, 1, 1)
        self.batchn2 = nn.BatchNorm2d(4*inc)

        self.sconv3 = nn.Conv2d(4*inc, 8*inc, 4, 2, 1)
        self.conv3 = nn.Conv2d(8*inc, 8*inc, 3, 1, 1)
        self.batchn3 = nn.BatchNorm2d(8*inc)

        self.sconv4 = nn.Conv2d(8*inc, 16*inc, 4, 2, 0)
        self.conv4 = nn.Conv2d(16*inc, 16*inc, 3, 1, 1)
        self.batchn4 = nn.BatchNorm2d(16*inc)

        self.sconv5 = nn.Conv2d(16*inc, 32*inc, 4, 2, 0)
        self.conv5 = nn.Conv2d(32*inc, 32*inc, 3, 1, 1)
        self.batchn5 = nn.BatchNorm2d(32*inc)

        self.conv6 = nn.Conv2d(32*inc, 1, 1, 1, 0)

        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.2, True)

    def discriminator(self, x):
        x = self.leakyrelu(self.batchn1(self.conv1(self.sconv1(x))))
        x = self.leakyrelu(self.batchn2(self.conv2(self.sconv2(x))))
        x = self.leakyrelu(self.batchn3(self.conv3(self.sconv3(x))))
        x = self.leakyrelu(self.batchn4(self.conv4(self.sconv4(x))))
        x = self.leakyrelu(self.batchn5(self.conv5(self.sconv5(x))))
        x = self.sigmoid(self.conv6(x))
        return x

    def forward(self, x):
        out = self.discriminator(x)
        return out.view(-1, 1).squeeze(1)
