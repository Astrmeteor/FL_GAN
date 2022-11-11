import os
import logging
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events
import ignite.distributed as idist
import ignite

ignite.utils.manual_seed(999)

ignite.utils.setup_logger(name="ignite.distributed.auto.auto_dataloader", level=logging.WARNING)
ignite.utils.setup_logger(name="ignite.distributed.launcher.Parallel", level=logging.WARNING)

my_device = ""

try:
    if torch.backends.mps.is_built():
        my_device = "mps"
except AttributeError:
    if torch.cuda.is_available():
        my_device = "cuda:0"
    else:
        my_device = "cpu"


class Flatten(nn.Module):
    """Flattens input by reshaping it into a one-dimensional tensor."""

    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    """Unflattens a tensor converting it to a desired shape."""

    def forward(self, input):
        return input.view(-1, 16, 6, 6)


class Net(nn.Module):
    def __init__(self, h_dim=576, z_dim=10) -> None:
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=6, kernel_size=4, stride=2
            ),  # [batch, 6, 15, 15]
            nn.ReLU(),
            nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=5, stride=2
            ),  # [batch, 16, 6, 6]
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=4, stride=2),
            nn.Tanh(),
        )

    def reparametrize(self, h):
        """Reparametrization layer of VAE."""
        mu, logvar = self.fc1(h), self.fc2(h)
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

    def encode(self, x):
        """Encoder of the VAE."""
        h = self.encoder(x)
        z, mu, logvar = self.reparametrize(h)
        return z, mu, logvar

    def decode(self, z):
        """Decoder of the VAE."""
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z_decode = self.decode(z)
        return z_decode, mu, logvar

def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    trainset = CIFAR10("/Users/tianying/Desktop/FL/data", train=True, download=True, transform=transform)
    testset = CIFAR10("/Users/tianying/Desktop/FL/data", train=False, download=True, transform=transform)

    num_examples = {"trainset_num": {len(trainset)}, "testset_num": {len(testset)}}
    return trainset, testset, num_examples


def train(engine, net, trainloader, valloader, epochs, device: str = "cpu"):
    optimizer = idist.auto_optim(
        optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    )
    net.train()
    LOSS = 0.0

    for images, _ in trainloader:
        optimizer.zero_grad()
        images = images.to(device)
        recon_images, mu, logvar = net(images)
        recon_loss = torch.nn.functional.mse_loss(recon_images, images)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.05 * kld_loss
        LOSS += loss
        loss.backward()
        optimizer.step()

    # fid = calculate_fid(images[:16].cpu().detach().numpy().reshape(16, -1),
    #                    recon_images[:16].cpu().detach().numpy().reshape(16, -1))
    """
    fid = calculate_fid_gpu(
        images.view(images.shape[0], -1),
        recon_images.view(recon_images.shape[0], -1)
    )
    """

    # train_fid = fid / len(images)
    # train_fid = fid / 16
    #train_loss = LOSS.detach() / len(trainloader.dataset)

    print("Starting validation_set test")
    # val_loss, val_fid = test(net, valloader, device=device)

    """
    if math.isnan(train_fid):
         train_fid = 0.0

    results = {
        "train_loss": float(train_loss),
        "train_fid": float(train_fid),
        "val_loss": float(val_loss),
        "val_fid": float(val_fid)
    }
    print("Training end ...")
    """
    # return results


if __name__ == "__main__":
    trainset, testset, _ = load_data()

    net = Net().to(my_device)
    n_valset = int(len(trainset) * 0.1)
    valset = torch.utils.data.Subset(trainset, range(0, n_valset))

    trainLoader = idist.auto_dataloader(trainset, batch_size=64, shuffle=True)
    valLoader = idist.auto_dataloader(valset, batch_size=64)

    # show the training data
    real_batch = next(iter(trainLoader))

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    # plt.show()

    net = idist.auto_model(Net())
    idist.device()
    # cifar size 3x32x32 CxHxW
    summary(net, (3, 32, 32))

    results = Engine(train)



