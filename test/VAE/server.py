import flwr as fl
from typing import Dict
from flwr.common import Metrics
from models import Net
import matplotlib.pyplot as plt
from flwr.common.typing import Scalar
import torch
import numpy as np
from client import CifarClient
from utils import set_parameters, load_data, test, load_partition
from torch.utils.data import DataLoader
import utils

def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
                 ):
        if server_round % 2 == 0:
            model = Net()
            model = set_parameters(model, parameters)  # Update model with the latest parameters
            random_vector_for_generation = torch.normal(0, 1, (16, 10))
            generate_and_save_images(model, server_round, random_vector_for_generation)

            _, testset, _ = load_data()
            valLoader = DataLoader(testset, batch_size=64)
            loss, fid = test(model, valLoader)
            return loss, {"fid": float(fid)}

    return evaluate


def generate_and_save_images(model, epoch, test_input):
    predictions = model.decode(test_input)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = np.transpose(predictions[i, :, :, ].detach().numpy(), axes=[1, 2, 0])
        plt.imshow((img/np.amax(img)*255).astype(np.uint8))
        plt.axis('off')
        # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('G_data/image_at_epoch_{:03d}.png'.format(epoch))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "local_epochs": 10,  # number of local epochs
        "batch_size": 64,
    }
    return config


def evaluate_fig(server_round: int):
    val_steps = 5 if server_round < 4 else 10
    val_batch_size = 64
    val_config = {
        "val_steps": val_steps,
        "val_batch_size": val_batch_size
    }

    return val_config


def client_fn(cid: str) -> CifarClient:
    net = Net().to(DEVICE)
    # trainset, testset, _ = load_data()
    trainset, testset = load_partition(int(cid))
    return CifarClient(cid, net, trainset, testset, DEVICE)


def main():

    s = fl.server.server.FedAvg(
        fraction_fit=1,
        fraction_evaluate=0.5,
        min_fit_clients=10,
        min_available_clients=10,
        min_evaluate_clients=2,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_fig
    )
    """
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=4), strategy=s,

    )
    """
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=s,
    )


if __name__ == "__main__":
    main()

    #net = Net()
    #train_loader, test_loader = utils.load_data()
    #utils.train(net, train_loader, 1)