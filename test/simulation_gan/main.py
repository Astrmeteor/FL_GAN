import argparse
import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
import torchvision
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, List
from dataset_utils import get_cifar_10, do_fl_partitioning, get_dataloader, load_data
from utils import train, test
from models.gan import Generator, Discriminator

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_client_cpus", type=int, default=2)
parser.add_argument("--num_rounds", type=int, default=1)


# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # Instantiate model
        self.generator = Generator()
        self.generator.initialize_weights()
        self.discriminator = Discriminator()
        self.discriminator.initialize_weights()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### obtain generator parameter
    def get_parameters(self, config):
        return get_params(self.generator)

    def fit(self, parameters, config):
        set_params(self.generator, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
        )

        # Send model to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Train
        train(self.generator, self.discriminator, trainloader, epochs=config["epochs"], device=self.device)

        # Return local model and statistics
        return get_params(self.generator), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        parameters = get_params(self.discriminator)
        net_d = Discriminator()
        set_params(net_d, parameters)

        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=64, workers=num_workers
        )

        # Send model to device
        self.discriminator.to(self.device)

        # Evaluate
        loss, accuracy = test(self.discriminator, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 3,  # number of local epochs
        "batch_size": 64,
    }
    return config


def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_evaluate_fn(
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
            parameters: fl.common.NDArrays,
    ):
        """Use the entire CIFAR-10 test set for evaluation."""
        ### generate synthetic data
        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Generator()
        set_params(model, parameters)
        model.to(device)

    return evaluate


if __name__ == "__main__":

    # parse input arguments
    args = parser.parse_args()

    pool_size = 2  # number of dataset partions (= number of total clients)
    num_client = 2
    client_resources = {
        "num_cpus": args.num_client_cpus
    }  # each client will get allocated 1 CPUs

    # Download CIFAR-10 dataset
    train_path, testset = get_cifar_10()

    fed_dir = do_fl_partitioning(
        train_path, pool_size=pool_size, alpha=1, num_classes=10, val_ratio=1
    )

    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=num_client,
        min_evaluate_clients=pool_size,
        min_available_clients=num_client,  # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_fn=None
        # evaluate_fn=get_evaluate_fn(),  # centralised evaluation of global model
    )


    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, fed_dir)


    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args
    )
    """
    trainloader, testloader, num_examples = load_data()
    g_net = Generator()
    g_net.initialize_weights()
    d_net = Discriminator()
    d_net.initialize_weights()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g_net.to(device)
    d_net.to(device)

    # train(g_net, d_net, trainloader, 2, device)
    loss, accuracy = test(d_net, testloader, device)
    print(loss, accuracy)
    """

