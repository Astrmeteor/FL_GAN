import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
from scipy import linalg
import tqdm
from pytorch_fid import fid_score
import pytorch_fid

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    trainset = CIFAR10("/Users/tianying/Desktop/FL/data", train=True, download=True, transform=transform)
    testset = CIFAR10("/Users/tianying/Desktop/FL/data", train=False, download=True, transform=transform)

    num_examples = {"trainset": {len(trainset)}, "testset": {len(testset)}}
    return trainset, testset, num_examples


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    trainset, testset, num_examples = load_data()
    n_train = int(num_examples["trainset"] / 10)
    n_test = int(num_examples["testset"] / 10)

    train_parition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_parition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )
    return (train_parition, test_parition)


def train(net, trainloader, valloader, epochs, device):
    """Train the network on the training set."""
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print("Starting training ...")
    net.train()
    for _ in range(epochs):
        for images, _ in trainloader:
            optimizer.zero_grad()
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images).to(device)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.05 * kld_loss
            loss.backward()
            optimizer.step()
    net.to("cpu")

    train_loss, train_fid = test(net, trainloader)
    val_loss, val_fid = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_fid": train_fid,
        "val_loss": val_loss,
        "val_fid": val_fid
    }

    return results


def test(net, testloader, steps: int = None, device: str = "cpu"):
    """Validate the network on the entire test set."""
    total, loss = 0, 0.0
    fid = 0
    print(f"Starting evaluation ({testloader.dataset.filename})...")
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in tqdm.tqdm(enumerate(testloader)):
            images = images.to(device)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images).to(device)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += recon_loss + kld_loss
            total += len(images)
            if steps is not None and batch_idx == steps:
                break
    fid = calculate_fid(images, recon_images)

    if steps is None:
        loss = loss / len(testloader.dataset),
        fid = sum(fid) / len(testloader.dataset)
    else:
        loss = loss / total,
        fid = fid / len(images)
    net.to("cpu")
    return loss, fid


def sample(net):
    """Generates samples using the decoder of the trained VAE."""
    with torch.no_grad():
        z = torch.randn(10)
        z = z.to(DEVICE)
        gen_image = net.decode(z)
    return gen_image


def generate(net, image):
    """Reproduce the input with trained VAE."""
    with torch.no_grad():
        return net.forward(image)


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def calculate_fid(act1, act2):
     # calculate mean and covariance statistics
     mu1 = act1.mean(axis=0)

     sigma1 = np.cov(act1, rowvar=False)

     mu2 = np.mean(act2, axis=0)

     sigma2 = np.cov(act2, rowvar=False)
     # mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
     # calculate sum squared difference between means

     ssdiff = np.sum((mu1 - mu2)**2.0)
     # calculate sqrt of product between cov
     covmean = linalg.sqrtm(sigma1.dot(sigma2))

     # check and correct imaginary numbers from sqrt
     if np.iscomplexobj(covmean):
      covmean = covmean.real
     # calculate score
     fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
     return fid

