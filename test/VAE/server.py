import flwr as fl
from typing import Tuple, Optional, Dict
from flwr.common import Metrics
from models import Net
import matplotlib.pyplot as plt
from flwr.common.typing import Scalar
import torch
import numpy as np

from utils import set_parameters

def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
                 ):
        if server_round % 2 == 0:
            model = Net()
            set_parameters(model, parameters)  # Update model with the latest parameters
            random_vector_for_generation = torch.randn(16, 10)
            generate_and_save_images(model, server_round, random_vector_for_generation)

    return evaluate


def generate_and_save_images(model, epoch, test_input):
    predictions = model.decode(test_input)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(np.transpose(predictions[i, :, :, ].detach().numpy(), axes=[1, 2, 0]))
        plt.axis('off')
        # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('G_data/image_at_epoch_{:03d}.png'.format(epoch))


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 2,  # number of local epochs
        "batch_size": 64,
    }
    return config


def main():
    NUM_CLIENTS = 3
    pool_size = 3
    
    s = fl.server.server.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_available_clients=pool_size,
        min_evaluate_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=fit_config,
    )
    fl.server.start_server(
        server_address="10.1.2.108:8080",
        config=fl.server.ServerConfig(num_rounds=20), strategy=s
    )


if __name__ == "__main__":
    main()