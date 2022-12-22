import math
import random
from models import VAE
from torch.utils.data import DataLoader

import flwr as fl
import torch
from utils import load_partition, Fl_Client, PARAMS


def main():

    dataset = ["mnist", "fashion-mnist", "cifar", "stl"]
    dataset = dataset[3]
    client_number = 2
    trainset, _ = load_partition(client_number, dataset)
    split = math.floor(len(trainset) * PARAMS["train_split"])
    client_trainset = torch.utils.data.Subset(trainset, list(range(0, split)))
    client_testset = torch.utils.data.Subset(trainset, list(range(split, len(trainset))))
    trainloader = DataLoader(client_trainset, PARAMS["batch_size"])
    testloader = DataLoader(client_testset, PARAMS["batch_size"])
    sample_rate = PARAMS["batch_size"] / len(client_trainset)
    model = VAE(dataset)
    my_device = ""

    try:
        if torch.backends.mps.is_built():
            my_device = "mps"
    except AttributeError:
        if torch.cuda.is_available():
            my_device = "cuda:0"
        else:
            my_device = "cpu"

    fl.client.start_numpy_client(server_address="10.1.2.102:8080",
                                 client=Fl_Client(cid=random.randint(1, 10),
                                                  model=model,
                                                  trainloader=trainloader,
                                                  testloader=testloader,
                                                  sample_rate=sample_rate,
                                                  device=my_device))


if __name__ == "__main__":
    # real situation
    main()