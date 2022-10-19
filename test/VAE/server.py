import flwr as fl
from typing import Tuple, Optional, Dict
from flwr.common import Metrics
from models import Net
import matplotlib.pyplot as plt
from flwr.common.typing import Scalar
import torch
import numpy as np
from client import CifarClient
from utils import set_parameters, load_data

import utils

def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
                 ):
        if server_round % 2 == 0:
            model = Net()
            set_parameters(model, parameters)  # Update model with the latest parameters
            random_vector_for_generation = torch.normal(0, 1, (16, 10))
            generate_and_save_images(model, server_round, random_vector_for_generation)

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
        "epochs": 2,  # number of local epochs
        "batch_size": 64,
    }
    return config

def client_fn(cid: str) -> CifarClient:
    net = Net().to(DEVICE)
    train_loader, test_loader = load_data()
    return CifarClient(net, train_loader, test_loader)

def main():
    NUM_CLIENTS = 1
    pool_size = 1
    
    s = fl.server.server.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=2,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=fit_config,
    )
    """
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=4), strategy=s,

    )
    """
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=s,
    )

if __name__ == "__main__":
    main()

    #net = Net()
    #train_loader, test_loader = utils.load_data()
    #utils.train(net, train_loader, 1)