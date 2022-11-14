import math
import random
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
from models import VAE
import time
import tqdm
import torchvision
my_device = ""

try:
    if torch.backends.mps.is_built():
        my_device = "mps"
except AttributeError:
    if torch.cuda.is_available():
        my_device = "cuda:0"
    else:
        my_device = "cpu"

DEVICE = torch.device(my_device)


def load_data(dataset):
    """Load (training and test set)."""
    data_transform = [transforms.ToTensor()]

    if dataset == "mnist":
        data_transform.append(transforms.Normalize((0.1307,), (0.3081,)))
        data_transform = transforms.Compose(data_transform)
        my_train = MNIST(root="./data", train=True, transform=data_transform, download=True)
        my_test = MNIST(root="./data", train=False, transform=data_transform, download=True)
    elif dataset == "fashion-mnist":
        data_transform.append(transforms.Normalize((0.5,), (0.5,)))
        data_transform = transforms.Compose(data_transform)
        my_train = FashionMNIST(root="./data", train=True, transform=data_transform, download=True)
        my_test = FashionMNIST(root="./data", train=False, transform=data_transform, download=True)
    elif dataset == "cifar":
        data_transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        data_transform = transforms.Compose(data_transform)
        my_train = CIFAR10(root="./data", train=True, transform=data_transform, download=True)
        my_test = CIFAR10(root="./data", train=False, transform=data_transform, download=True)
    elif dataset == "stl":
        data_transform.append(transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]))
        data_transform = transforms.Compose(data_transform)
        my_train = STL10(root="./data", split="unlabeled", transform=data_transform, download=True)
        my_test = STL10(root="./data", split="test", transform=data_transform, download=True)

    return my_train, my_test


def load_partition(idx: int, my_dataset):
    """Load idx th of the training and test data to simulate a partition."""
    """idx means how many clients will join the learning, so that we split related subsets."""
    # assert idx in range(10)
    trainset, testset = load_data(my_dataset)

    # Each number of dataset
    n_train = int(len(trainset) / idx)
    n_test = int(len(testset) / idx)

    # Random choice one piece to load
    idx_start = random.randint(0, idx-1)

    bound_train = ((idx_start + 1) * n_train) if ((idx_start + 1) * n_train) < len(trainset) else len(trainset)
    bound_test = (idx_start + 1) * n_test if ((idx_start + 1) * n_test) < len(testset) else len(testset)

    train_parition = torch.utils.data.Subset(
        trainset, range(idx_start * n_train, bound_train)
    )
    test_parition = torch.utils.data.Subset(
        testset, range(idx_start * n_test, bound_test)
    )

    return (train_parition, test_parition)

features_out_hook = []
def layer_hook(module, inp, out):
    features_out_hook.append(out.data.cpu().numpy())

def train(net, trainloader, valloader, epochs, dp: str = "", device: str = "cpu"):
    """Train the network on the training set."""
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # print("Starting training ...")
    # print(f"DEVICE: {device}")
    net.train()
    LOSS = 0.0

    train_fid = 0.0
    start_time = time.time()

    for e in tqdm.tqdm(range(epochs)):
        loop = tqdm.tqdm((trainloader), total=len(trainloader), leave=False)
        for images, labels in loop:

            optimizer.zero_grad()

            if dp == "laplace":
                images = Laplace_mech(images).float()
            if dp == "gaussian":
                images = Gaussian_mech(images)

            images = images.to(device)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.05 * kld_loss
            LOSS += loss
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{e}/{epochs}]")
            loop.set_postfix(loss=loss.item())

    # Calculate FID

    train_fid = compute_fid(images, recon_images, device)

    train_loss = LOSS.detach() / len(trainloader.dataset)

    # print("Starting validation_set test")
    val_loss, val_fid = test(net, valloader, device=device)


    results = {
        "train_loss": float(train_loss),
        "fid": float(train_fid),
        "val_loss": float(val_loss),
        "val_fid": float(val_fid)
    }
    # print("Training end ...")
    return results


def test(net, testloader, dp : str = "", device: str = "cpu"):
    """Validate the network on the entire test set."""
    loss = 0.0
    # print("Starting evaluation ...")
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):

            if dp == "laplace":
                images = Laplace_mech(images).to(torch.float32)
            if dp == "gaussian":
                images = Gaussian_mech(images).to(torch.float32)

            images = images.to(device)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += recon_loss + kld_loss

    loss = loss.detach() / len(testloader.dataset)
    # fid = fid / len(images)
    fid = compute_fid(images, recon_images, device)

    return loss, fid


def compute_fid(images, recon_images, device):
    torch_size = torchvision.transforms.Resize([299, 299])

    images_resize = torch_size(images).to(device)

    fid_model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
    fid_model.to(device)

    global features_out_hook
    features_out_hook = []
    hook1 = fid_model.dropout.register_forward_hook(layer_hook)
    fid_model.eval()
    fid_model(images_resize)
    images_features = np.array(features_out_hook).squeeze()
    hook1.remove()

    features_out_hook = []
    recon_images_resize = torch_size(recon_images).to(device)
    hook2 = fid_model.dropout.register_forward_hook(layer_hook)
    fid_model.eval()
    fid_model(recon_images_resize)
    recon_images_features = np.array(features_out_hook).squeeze()
    hook2.remove()

    fid = calculate_fid(images_features, recon_images_features)

    return fid


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
     ssdiff = np.sum((mu1 - mu2)**2)

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

     return fid


def Laplace_mech(data, epsilon: float = 1.0):
    loc = 0
    sensitive = 1
    scale = sensitive / epsilon
    s = np.random.laplace(loc, scale, data.shape)
    data = data + s
    return data


def Gaussian_mech(data, epsilon: float = 1.0):
    loc = 0
    sensitive = 1
    delta = 1 / pow(len(data.data), 2) if pow(len(data.data), 2) > 10e6 else 10e-6
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitive / epsilon
    s = np.random.normal(loc, sigma, data.shape)
    data.data = data + s
    return data