import flwr as fl
from typing import Dict, List, Tuple, Union, Optional
from flwr.common import Metrics
from models import VAE
import matplotlib.pyplot as plt
from flwr.common.typing import Scalar
import torch
import numpy as np
from client import CifarClient
from utils import load_data, test, load_partition
from torch.utils.data import DataLoader
from collections import OrderedDict


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            if server_round % 2 == 0:
                # Save aggregated_ndarrays
                print(f"Saving round {server_round} aggregated_ndarrays...")
                np.savez(f"metrics/round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


def get_evaluate_fn(model: torch.nn.Module):
    """Return an evaluation function for server-side evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
                 ):

        # model = Net()
        # model = set_parameters(model, parameters)  # Update model with the latest parameters
        my_device = ""

        try:
            if torch.backends.mps.is_built():
                my_device = "mps"
        except AttributeError:
            if torch.cuda.is_available():
                my_device = "cuda:0"
            else:
                my_device = "cpu"

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(my_device)
        if server_round % 2 == 0:

            dataset = ["mnist", "fashion-mnist", "cifar", "stl"]
            _, testset = load_partition(2, dataset[3])

            valLoader = DataLoader(testset, batch_size=64, shuffle=True)
            images = next(iter(valLoader))
            recon_images, _, _ = model(images[0].to(my_device))

            predictions = recon_images[:16]
            fig = plt.figure(figsize=(3, 3))
            for i in range(predictions.shape[0]):
                plt.subplot(4, 4, i + 1)
                img = np.transpose(predictions[i, :, :, ].cpu().detach().numpy(), axes=[1, 2, 0])
                # plt.imshow((img/np.amax(img)*255).astype(np.uint8))
                plt.imshow(img)
                plt.axis('off')
                # tight_layout minimizes the overlap between 2 sub-plots
            plt.savefig('FL_G_data/image_at_epoch_{:03d}.png'.format(server_round))
            plt.close()


        # dataset = config["dataset"]
        dataset = "stl"
        _, testset = load_data(dataset)
        valLoader = DataLoader(testset, batch_size=64)

        my_device = ""

        try:
            if torch.backends.mps.is_built():
                my_device = "mps"
        except AttributeError:
            if torch.cuda.is_available():
                my_device = "cuda:0"
            else:
                my_device = "cpu"

        model.to(my_device)
        print(f"Global validation starting with device: {my_device}...")
        loss, fid = test(model, valLoader, device=my_device)
        print(f"Global validation end with FID: {fid}")
        return loss, {"fid": float(fid)}

    return evaluate


def generate_and_save_images(model, epoch, test_input):
    predictions = model.decoder(model.fc3(test_input))
    fig = plt.figure(figsize=(3, 3))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = np.transpose(predictions[i, :, :, ].detach().numpy(), axes=[1, 2, 0])
        # plt.imshow((img/np.amax(img)*255).astype(np.uint8))
        plt.imshow(img)
        plt.axis('off')
        # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('G_data/image_at_epoch_{:03d}.png'.format(epoch))
    plt.close()


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    dp = ["laplace", "gaussian"]
    config = {
        "local_epochs": 10,  # number of local epochs
        "batch_size": 64,
        "server_round": server_round,
        "dp": dp
    }
    return config


def evaluate_fig(server_round: int):
    val_batch_size = 64
    dataset = ["mnist", "fashion-mnist", "cifar", "stl"]
    val_config = {
        "val_batch_size": val_batch_size,
        "server_round": server_round,
        "dataset": dataset[3]
    }

    return val_config


def client_fn(cid: str) -> CifarClient:
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

    # net = Net().to(DEVICE)
    # trainset, testset, _ = load_data()

    dataset = ["mnist", "fashion-mnist", "cifar", "stl"]
    trainset, testset = load_partition(4, dataset[3])
    return CifarClient(cid, trainset, testset, device=DEVICE)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    fid = [m["fid"] for number, m in metrics]
    fid = np.average(fid)
    # Aggregate and return average fid
    return {"fid": fid}


def main():
    num_client = 2

    dataset_type = "stl"
    model = VAE(dataset_type)

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    strategy = SaveModelStrategy(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=num_client,
        min_available_clients=num_client,
        min_evaluate_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_fig,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        evaluate_metrics_aggregation_fn=weighted_average
    )
    """
    fl.server.start_server(
        server_address="10.1.2.102:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy
    )
    """

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_client,
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy
    )


if __name__ == "__main__":
    main()