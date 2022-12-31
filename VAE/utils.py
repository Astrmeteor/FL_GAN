import random
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
import tqdm
import torchvision
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import flwr as fl
from typing import OrderedDict

"""
PARAMS = {
    "batch_size": 64,
    "train_split": 0.7,
    "local_epochs": 1
}

PRIVACY_PARAMS = {
    "target_delta": 1e-05,
    "noise_multiplier": 0.4,
    "max_grad_norm": 1.2,
    "target_epsilon": 50,
    "max_batch_size": 128
}
"""

def get_labels(dataset_name: str = "stl"):
    if dataset_name == "stl":
        label_names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    elif dataset_name == "cifar":
        label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif dataset_name == "fashion-mnist":
        label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                       "Ankle boot"]
    elif dataset_name == "mnist":
        label_names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    return label_names


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
    idx_start = random.randint(0, idx - 1)

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


def train(net, trainloader, epochs, privacy_engine, device: str = "cpu"):
    """Train the network on the training set."""
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-03, weight_decay=1e-05)
    # privacy_engine.attach(optimizer)

    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=net,
        optimizer=optimizer,
        data_loader=trainloader,
        epochs=epochs,
        target_epsilon=PARAMS["target_epsilon"],
        target_delta=PARAMS["target_delta"],
        max_grad_norm=PARAMS["max_grad_norm"],
    )

    # print("Starting training ...")
    # print(f"DEVICE: {device}")

    LOSS = 0.0

    train_fid = 0.0
    for e in tqdm.tqdm(range(epochs)):

        with BatchMemoryManager(
                data_loader=trainloader,
                max_physical_batch_size=PARAMS["max_batch_size"],
                optimizer=optimizer
        ) as memory_safe_data_loader:

            loop = tqdm.tqdm((memory_safe_data_loader), total=len(trainloader), leave=False)
            for images, labels in loop:
                # if loop.last_print_n == 2:
                #    break
                optimizer.zero_grad()

                images = images.to(device)
                recon_images, mu, logvar = model(images)
                recon_loss = F.mse_loss(recon_images, images)
                kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kld_loss
                LOSS += loss
                loss.backward()
                optimizer.step()
                loop.set_description(f"Epoch [{e}/{epochs}]")
                loop.set_postfix(loss=loss.item())

                # Gradient clipping
                # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)

    epsilon, _ = optimizer.privacy_engine.get_privacy_spent(PRIVACY_PARAMS["target_delta"])
    # Calculate FID
    if images.shape[1] == 1:
        images = torch.repeat_interleave(images, repeats=3, dim=1)
        recon_images = torch.repeat_interleave(recon_images, repeats=3, dim=1)

    train_fid = compute_fid(images, recon_images, device)

    train_loss = LOSS.detach().item()

    results = {
        "train_loss": float(train_loss),
        "fid": float(train_fid),
        "epsilon": epsilon
    }
    return results


def test(net, testloader, device: str = "cpu"):
    """Validate the network on the entire test set."""
    loss = 0.0
    # print("Starting evaluation ...")
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images = images.to(device)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += recon_loss + kld_loss

    loss = loss.detach().item()
    # fid = fid / len(images)
    if images.shape[1] == 1:
        images = torch.repeat_interleave(images, repeats=3, dim=1)
        recon_images = torch.repeat_interleave(recon_images, repeats=3, dim=1)

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


# def set_parameters(model, parameters):
#    params_dict = zip(model.state_dict().keys(), parameters)
#    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#    model.load_state_dict(state_dict, strict=True)
#    return model


def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def get_labels(dataset_name: str = "stl"):
    if dataset_name == "stl":
        label_names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    elif dataset_name == "cifar":
        label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif dataset_name == "fashion-mnist":
        label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                       "Ankle boot"]
    elif dataset_name == "mnist":
        label_names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    return label_names


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1 = act1.mean(axis=0)

    sigma1 = np.cov(act1, rowvar=False)

    mu2 = np.mean(act2, axis=0)

    sigma2 = np.cov(act2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2)

    # calculate sqrt of product between cov
    mual = sigma1.dot(sigma2)
    # covmean = linalg.sqrtm(mual)
    covmean = linalg.sqrtm(mual)

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    trace = np.trace(sigma1 + sigma2 - 2.0 * covmean)
    fid = ssdiff + trace

    return fid


class Fl_Client(fl.client.NumPyClient):
    def __init__(
            self,
            cid,
            model,
            # trainset: torchvision.datasets,
            # testset: torchvision.datasets,
            trainloader: DataLoader,
            testloader: DataLoader,
            sample_rate: float,
            device: str
            # validation_split: int = 0.1
    ) -> None:
        super().__init__()
        self.cid = cid
        self.model = model
        # self.train = trainset
        # self.testset = testset
        self.trainloader = trainloader
        self.tesloader = testloader
        self.device = device

        # Create a privacy engine which will add DP and keep track of the privacy budget
        self.privacy_engine = PrivacyEngine()

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict[{k: torch.tensor(v) for k, v in params_dict}]
        self.model.load_state_dict(state_dict, strict=True)
        # return model

    def get_parameters(self):
        # print(f"[Client {self.cid}] get_parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        print(f"Global epoch {config['server_round']}: [Client {self.cid}] fit, config: {config}")
        # model = self.set_parameters(parameters)

        self.set_parameters(parameters)

        # batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # trainset = torch.utils.data.Subset(self.trainset, range(n_valset, len(self.trainset)))
        # trainset = self.trainset
        print(f"Client {self.cid}: train dataset number {len(self.trainloader)}, Starting training ...")
        # trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
        # valLoader = DataLoader(valset, batch_size=batch_size)

        self.model.to(self.device)

        results = train(self.model, self.trainLoader, epochs, self.privacy_engine, self.device)
        print(f"Client {self.cid}: Train FID {results['fid']}, Epsilon {results['epsilon']}. Training end ...")
        # parameters_prime = get_model_params(model)
        # num_examples = len(trainLoader)

        # return parameters_prime, num_examples, results
        return self.get_parameters(), len(self.trainloader), results

    def evaluate(self, parameters, config):
        print(f"Global epoch {config['server_round']}: [Client {self.cid}] evaluate, config: {config}")
        # model = self.set_parameters(parameters)
        # testloader = DataLoader(self.testset, batch_size=config["val_batch_size"])

        # n_valset = int(len(self.trainset) * self.validation_split)
        # valset = torch.utils.data.Subset(self.trainset, range(0, n_valset))

        self.model.to(self.device)
        print(f"Client {self.cid}: test dataset size {len(self.tesloader)}, Starting validation ...")
        loss, fid = test(self.model, self.tesloader, device=self.device)

        print(f"Client {self.cid}: Test FID: {fid}, Test loss: {loss}. Validation end ...")

        return float(loss), len(self.tesloader), {"fid": float(fid)}
