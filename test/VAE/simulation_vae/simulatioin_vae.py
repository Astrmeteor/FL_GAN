import flwr as fl
from typing import Tuple, Optional, Dict
from flwr.common import Metrics
import matplotlib.pyplot as plt
from flwr.common.typing import Scalar
import torch
import numpy as np
from common import set_parameters, train, test, Net
from dataset_utils import get_cifar_10, do_fl_partitioning, get_dataloader

class CifarClient(fl.client.NumPyClient):
    def __init__(
            self,
            model: Net,
            trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.fid = 0

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    """
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    """

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.fid = train(self.model, self.trainloader, epochs=config["epochs"])
        return self.get_parameters(config), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss = test(self.model, self.testloader)
        return float(self.fid), len(self.testloader), {}


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
        plt.imshow((img / np.amax(img) * 255).astype(np.uint8))
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
    pool_size = 5  # number of dataset partions (= number of total clients)
    num_client = 4  # each epoch client number
    client_resources = {
        "num_cpus": 2
    }

    # Download CIFAR-10 dataset
    train_path, testset = get_cifar_10()

    fed_dir = do_fl_partitioning(
        train_path, pool_size=pool_size, alpha=1, num_classes=10, val_ratio=1
    )

    strategy = fl.server.server.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_available_clients=pool_size,
        min_evaluate_clients=num_client,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=fit_config,
    )

    def client_fn(cid: str):
        # create a single client instance
        return CifarClient(cid, fed_dir)

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        ray_init_args=ray_init_args
    )


if __name__ == "__main__":
    main()