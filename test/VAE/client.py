
from models import Net
from torch.utils.data import DataLoader
from utils import *
import flwr as fl
import torchvision


class CifarClient(fl.client.NumPyClient):
    def __init__(
            self,
            cid,
            model: Net,
            trainset: torchvision.datasets,
            testset: torchvision.datasets,
            device: str,
            validation_split: int = 0.1,
    ) -> None:
        self.cid = cid
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.device = device,
        self.validation_split = validation_split

    """
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    """

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        model = set_parameters(self.model, parameters)
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        n_valset = int(len(self.trainset) * self.validation_split)
        valset = torch.utils.data.Subset(self.trainset, range(0, n_valset))

        trainset = torch.utils.data.Subset(
            self.trainset, range(n_valset, len(self.trainset))
        )
        print(f"Clinet {int(self.cid)}: dataset number {len(trainset)}")
        trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valLoader = DataLoader(valset, batch_size=batch_size)

        results = train(model, trainLoader, valLoader, epochs, self.device)

        parameters_prime = self.get_parameters()
        num_examples = len(trainset)

        return parameters_prime, num_examples, results

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        model = set_parameters(self.model, parameters)
        steps : int = config["val_steps"]
        testloader = DataLoader(self.testset, batch_size=config["val_batch_size"])
        loss, fid = test(model, testloader)

        return float(loss), len(self.testset), {"fid": float(fid)}


def main():
    # Load model and data
    net = Net()
    trainloader, testloader = load_data()

    fl.client.start_numpy_client(server_address="localhost:8080",
                                 client=CifarClient(net, trainloader, testloader))


if __name__ == "__main__":
    main()