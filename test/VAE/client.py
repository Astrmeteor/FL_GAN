
from models import Net
from torch.utils.data import DataLoader
from utils import *
import flwr as fl


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
        train(self.model, self.trainloader, epochs=config["epochs"])
        return self.get_parameters(config), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss = test(self.model, self.testloader)
        return float(loss), len(self.testloader), {}


def main():
    # Load model and data
    net = Net()
    trainloader, testloader = load_data()

    fl.client.start_numpy_client(server_address="localhost:8080",
                                 client=CifarClient(net, trainloader, testloader))


if __name__ == "__main__":
    main()