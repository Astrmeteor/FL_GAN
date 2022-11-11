import flwr as fl
import cifar
import torch
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from collections import OrderedDict

DEVICE = str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

def fit_config(server_round: int):
    config = {
        "current_round": server_round
    }

    return config

net = cifar.Net().to(DEVICE)

class SaveModelStrategy(fl.server.server.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_manager.ClientProxy, fl.server.server.FitRes]],
        failures: List[Union[Tuple[fl.server.client_manager.ClientProxy, fl.server.server.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.server.server.Parameters], Dict[str, fl.server.server.Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(net.state_dict(), f"model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics


if __name__ == "__main__":
    NUM_CLIENTS = 2
    s = SaveModelStrategy(
        fraction_fit=1,
        fraction_evaluate=1,
        min_available_clients=2,
        min_evaluate_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config
    )
    fl.server.start_server(server_address="10.1.2.122:8080", config=fl.server.ServerConfig(num_rounds=3), strategy=s)

