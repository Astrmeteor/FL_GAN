import flwr as fl
from typing import Dict
from flwr.common import Metrics
from models import Net
import matplotlib.pyplot as plt
from flwr.common.typing import Scalar
import torch
import numpy as np
from client import CifarClient
from utils import load_data, test, load_partition
from torch.utils.data import DataLoader
from collections import OrderedDict


def get_evaluate_fn(model: torch.nn.Module):
    """Return an evaluation function for server-side evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
                 ):

        # model = Net()
        # model = set_parameters(model, parameters)  # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        if server_round % 2 == 0:
            random_vector_for_generation = torch.normal(0, 1, (16, 10))
            generate_and_save_images(model, server_round, random_vector_for_generation)

        _, testset, _ = load_data()
        valLoader = DataLoader(testset, batch_size=64)
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss, fid = test(model, valLoader, device=DEVICE)
        return loss, {"fid": float(fid)}

    return evaluate


def generate_and_save_images(model, epoch, test_input):
    predictions = model.decode(test_input)
    fig = plt.figure(figsize=(3, 3))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = np.transpose(predictions[i, :, :, ].detach().numpy(), axes=[1, 2, 0])
        plt.imshow((img/np.amax(img)*255).astype(np.uint8))
        plt.axis('off')
        # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('G_data/image_at_epoch_{:03d}.png'.format(epoch))



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
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = Net().to(DEVICE)
    # trainset, testset, _ = load_data()
    trainset, testset = load_partition(int(cid))
    return CifarClient(cid, trainset, testset, device=DEVICE)


def main():
    num_client = 2

    model = Net()

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    s = fl.server.server.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=num_client,
        min_available_clients=num_client,
        min_evaluate_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_fig,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters)
    )

    fl.server.start_server(
        server_address="10.1.2.102:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=s
    )
    """

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_client,
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=s,
    )
    """


if __name__ == "__main__":
    main()