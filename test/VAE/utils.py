import random

import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
from models import Net
import time

my_device = ""

if torch.backends.mps.is_built():
    my_device = "mps"
elif torch.cuda.is_available():
    my_device = "cuda:0"
else:
    my_device = "cpu"
DEVICE = torch.device(my_device)


def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    trainset = CIFAR10("/Users/tianying/Desktop/FL/data", train=True, download=True, transform=transform)
    testset = CIFAR10("/Users/tianying/Desktop/FL/data", train=False, download=True, transform=transform)

    num_examples = {"trainset_num": {len(trainset)}, "testset_num": {len(testset)}}
    return trainset, testset, num_examples


def load_partition(idx: int):
    """Load idx th of the training and test data to simulate a partition."""
    """idx means how many clients will join the learning, so that we split related subsets."""
    # assert idx in range(10)
    trainset, testset, num_examples = load_data()
    n_train = int(len(trainset) / idx)
    n_test = int(len(testset) / idx)

    idx_start = random.randint(0, idx)

    bound_train = ((idx_start + 1) * n_train) if ((idx_start + 1) * n_train) > len(trainset) else len(trainset)
    bound_test = (idx_start + 1) * n_test if ((idx_start + 1) * n_test) > len(testset) else len(testset)

    train_parition = torch.utils.data.Subset(
        trainset, range(idx_start * n_train, bound_train)
    )
    test_parition = torch.utils.data.Subset(
        testset, range(idx_start * n_test, bound_test)
    )
    return (train_parition, test_parition)


def train(net, trainloader, valloader, epochs, device: str = "cpu"):
    """Train the network on the training set."""
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print("Starting training ...")
    print(f"DEVICE: {device}")
    net.train()
    LOSS = 0.0

    for _ in range(epochs):
        for images, _ in trainloader:
            optimizer.zero_grad()
            images = images.to(device)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.05 * kld_loss
            LOSS += loss
            loss.backward()
            optimizer.step()
    # net.to("cpu")
    print("Starting train_set test")
    # fid = calculate_fid(images.numpy().reshape(images.size(0), -1),
    #                   recon_images.numpy().reshape(recon_images.size(0), -1))
    fid = calculate_fid(images.cpu().detach().numpy().reshape(images.size(0), -1), recon_images.cpu().detach().numpy().reshape(recon_images.size(0), -1))

    train_fid = fid / len(images)
    train_loss = LOSS.detach() / len(trainloader.dataset)

    print("Starting validation_set test")
    val_loss, val_fid = test(net, valloader, device=device)

    results = {
        "train_loss": float(train_loss),
        "train_fid": float(train_fid),
        "val_loss": float(val_loss),
        "val_fid": float(val_fid)
    }
    print("Training end ...")
    return results


def test(net, testloader, steps: int = None, device="cpu"):
    """Validate the network on the entire test set."""
    total, loss = 0, 0.0
    fid = 0
    print("Starting evaluation ...")
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images = images.to(device)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images).to(device)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += recon_loss + kld_loss
            total += len(images)
            if steps is not None and batch_idx == steps:
                break

    fid = calculate_fid(images.cpu().detach().numpy().reshape(images.size(0), -1), recon_images.cpu().detach().numpy().reshape(recon_images.size(0), -1))
    print(f"FID: {fid/len(images)}")
    if steps is None:
        loss = loss.detach() / len(testloader.dataset)
        fid = fid / len(images)
    else:
        loss = loss.detach() / total
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


# def set_parameters(model, parameters):
#    params_dict = zip(model.state_dict().keys(), parameters)
#    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#    model.load_state_dict(state_dict, strict=True)
#    return model

def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def calculate_fid(act1, act2):
     # calculate mean and covariance statistics
     mu1 = act1.mean(axis=0)

     sigma1 = np.cov(act1, rowvar=False)

     mu2 = np.mean(act2, axis=0)

     sigma2 = np.cov(act2, rowvar=False)


     # calculate sum squared difference between means
     ssdiff = np.sum((mu1 - mu2)**2.0)

     # calculate sqrt of product between cov
     mual = sigma1.dot(sigma2)
     #covmean = linalg.sqrtm(mual)
     covmean = linalg.sqrtm(mual)

     # check and correct imaginary numbers from sqrt
     if np.iscomplexobj(covmean):
      covmean = covmean.real
     # calculate score
     trace = np.trace(sigma1 + sigma2 - 2.0 * covmean)
     fid = ssdiff + trace

     cpu_result = {
         "mu1": mu1,
         "sigma1": sigma1,
         "mu2": mu2,
         "sigma2": sigma2,
         "ssdiff": ssdiff,
         "mual": mual,
         "covmean": covmean,
         "trace": trace,
         "fid": fid
     }
     return fid, cpu_result


def calculate_fid_gpu(act1, act2):
    # calculate mean and covariance statistics

    mu1 = torch.mean(act1, dim=0)
    sigma1 = torch.cov(act1.transpose(0, 1))

    mu2 = torch.mean(act2, dim=0)

    sigma2 = torch.cov(act2.transpose(0, 1))

    # mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means

    ssdiff = torch.sum((mu1 - mu2) ** 2.0)

    # calculate sqrt of product between cov

    mual = torch.mm(sigma1, sigma2)
    covmean = torch.sqrt(mual + 1e-8)

    trace = sigma1 + sigma2 - 2.0 * covmean
    trace = torch.diagonal(trace, 0)
    trace = torch.sum(trace)
    # trace = torch.trace(sigma1 + sigma2 - 2.0 * covmean)

    # calculate score
    fid = ssdiff + trace

    gpu_result = {
        "mu1": mu1,
        "sigma1": sigma1,
        "mu2": mu2,
        "sigma2": sigma2,
        "ssdiff": ssdiff,
        "mual": mual,
        "covmean": covmean,
        "trace": trace,
        "fid": fid
    }

    return fid, gpu_result


if __name__ == "__main__":
    """
    trainset, testset, _ = load_data()
    net = Net().to(DEVICE)
    n_valset = int(len(trainset) * 0.1)
    valset = torch.utils.data.Subset(trainset, range(0, n_valset))

    trainLoader = DataLoader(trainset, batch_size=64, shuffle=True)
    valLoader = DataLoader(valset, batch_size=64)

    results = train(net, trainLoader, valLoader, 1, DEVICE)
    print(results)
    """

    x = np.random.random((60, 1000)).astype(np.float32)
    y = np.random.random((60, 1000)).astype(np.float32)

    s_1 = time.time()
    cpu_fid, c_s = calculate_fid(x, y)
    print(f"CPU FID: {cpu_fid} time: {time.time() - s_1}")

    s_2 = time.time()
    gpu_fid, g_s = calculate_fid_gpu(torch.tensor(x, ).to("mps"), torch.tensor(y).to("mps"))
    print(f"GPU FID: {gpu_fid} time: {time.time() - s_2}")

