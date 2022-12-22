import flwr as fl
from typing import Dict, List, Tuple, Union, Optional
from flwr.common import Metrics
from models import VAE
import matplotlib.pyplot as plt
from flwr.common.typing import Scalar
import torch
import numpy as np
from client import Fl_Client
from utils import test, load_partition, PARAMS
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
        global DATASET, CLIENT, DP
        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")

            np.savez(f"Results/{DATASET}/{CLIENT}/{DP}/weights/FL_round_{server_round}_weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


def get_evaluate_fn(model: torch.nn.Module):
    """Return an evaluation function for server-side evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
                 ):

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

        global DATASET, CLIENT, DP
        _, testset = load_partition(2, DATASET)

        valLoader = DataLoader(testset, batch_size=64, shuffle=True)
        images, labels = next(iter(valLoader))
        recon_images, _, _ = model(images.to(my_device))

        ground_truth = images[:16]
        predictions = recon_images[:16]
        fig = plt.figure(figsize=(3, 3))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            img = np.transpose(predictions[i, :, :, ].cpu().detach().numpy(), axes=[1, 2, 0])
            # img = img / 2 + 0.5
            plt.imshow(img)
            plt.axis('off')

        plt.savefig('Results/{}/{}/{}/FL_G_data/image_at_epoch_{:03d}_G.png'.format(DATASET, CLIENT, DP, server_round))
        plt.close()

        fig2 = plt.figure(figsize=(3, 3))
        for i in range(ground_truth.shape[0]):
            plt.subplot(4, 4, i + 1)
            img = np.transpose(ground_truth[i, :, :, ].cpu().detach().numpy(), axes=[1, 2, 0])

            plt.imshow(img)
            plt.axis('off')

        plt.savefig('Results/{}/{}/{}/FL_G_data/image_at_epoch_{:03d}.png'.format(DATASET, CLIENT, DP, server_round))
        plt.close()

        label_names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
        val_labels = [label_names[i] for i in np.array(labels)]

        np.savetxt("Results/{}/{}/{}/Labels/label_at_epoch_{:03d}.txt".format(DATASET, CLIENT, DP, server_round), labels, fmt="%s")

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


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    global DP
    config = {
        "local_epochs": 10,  # number of local epochs
        "batch_size": 64,
        "server_round": server_round,
        "dp": DP
    }
    return config


def evaluate_fig(server_round: int):
    val_batch_size = 64
    global DATASET, DP
    dataset = DATASET
    dp = DP
    val_config = {
        "val_batch_size": val_batch_size,
        "server_round": server_round,
        "dataset": dataset,
        "dp": dp
    }

    return val_config


def client_fn(cid: str) -> Fl_Client:
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

    global DATASET, CLIENT
    model = VAE(DATASET)

    trainset, testset = load_partition(CLIENT, DATASET)
    client_trainloader = DataLoader(trainset, PARAMS["batch_size"])
    client_testloader = DataLoader(test, PARAMS["batch_size"])
    sample_rate = PARAMS["batch_size"] / len(trainset)
    return Fl_Client(cid, client_trainloader, client_testloader, device=DEVICE)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply metrics of each client by number of examples used
    # fid = [m["fid"] for number, m in metrics]
    # fid = np.average(fid)
    # Aggregate and return average fid
    met = {}
    for i, m in enumerate(metrics):
        met[i] = m[1]["fid"]
    #return {"fid": fid}
    return met


DATASET = ["mnist", "fashion-mnist", "cifar", "stl"]
DATASET = DATASET[3]
DP = ["normal", "laplace", "gaussian"]
DP = DP[0]
CLIENT = [1, 2, 3, 4, 5]
CLIENT = CLIENT[1]

def main():
    num_client = CLIENT

    model = VAE(DATASET)

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    strategy = SaveModelStrategy(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=num_client,
        min_available_clients=num_client,
        min_evaluate_clients=num_client,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_fig,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        evaluate_metrics_aggregation_fn=weighted_average
    )


    history = fl.server.History()
    num_rounds = 100
    history = fl.server.start_server(
        server_address="10.1.2.102:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

    # print(history)
    np.save(f"Results/{DATASET}/{CLIENT}/{DP}/metrics/fid_loss_{num_rounds}.npy", history)


if __name__ == "__main__":
    # real situation
    main()