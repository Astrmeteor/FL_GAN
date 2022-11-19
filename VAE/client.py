import random

from models import VAE
from torch.utils.data import DataLoader
from utils import train, test
import flwr as fl
import torchvision
import torch
from collections import OrderedDict
from utils import get_model_params, load_partition


class CifarClient(fl.client.NumPyClient):
    def __init__(
            self,
            cid,
            # model: Net,
            trainset: torchvision.datasets,
            testset: torchvision.datasets,
            device: str
            # validation_split: int = 0.1
    ) -> None:
        self.cid = cid
        # self.model = model
        self.trainset = trainset
        self.testset = testset
        self.device = device
        # self.validation_split = validation_split

    def set_parameters(self, parameters):
        model = VAE("stl")
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model

    # def get_parameters(self, config):
    #    print(f"[Client {self.cid}] get_parameters")
    #    return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        print(f"Global epoch {config['server_round']}: [Client {self.cid}] fit, config: {config}")
        model = self.set_parameters(parameters)
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # trainset = torch.utils.data.Subset(self.trainset, range(n_valset, len(self.trainset)))
        trainset = self.trainset
        print(f"Client {self.cid}: train dataset number {len(trainset)}, Starting training ...")
        trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
        # valLoader = DataLoader(valset, batch_size=batch_size)

        model.to(self.device)
        results = train(model, trainLoader, epochs, config["dp"], self.device)
        print(f"Client {self.cid}: Train FID: {results['fid']}. Training end ...")
        parameters_prime = get_model_params(model)
        num_examples = len(trainLoader)

        return parameters_prime, num_examples, results

    def evaluate(self, parameters, config):
        print(f"Global epoch {config['server_round']}: [Client {self.cid}] evaluate, config: {config}")
        model = self.set_parameters(parameters)
        testloader = DataLoader(self.testset, batch_size=config["val_batch_size"])

        # n_valset = int(len(self.trainset) * self.validation_split)
        # valset = torch.utils.data.Subset(self.trainset, range(0, n_valset))

        model.to(self.device)
        print(f"Client {self.cid}: test dataset size {len(testloader)}, Starting validation ...")
        loss, fid = test(model, testloader, config["dp"], device=self.device)

        print(f"Client {self.cid}: Test FID: {fid}, Test loss: {loss}. Validation end ...")

        return float(loss), len(self.testset), {"fid": float(fid)}


def main():
    # Load model and data
    # net = Net()
    # dp = "laplace"
    dataset = ["mnist", "fashion-mnist", "cifar", "stl"]
    trainset, testset = load_partition(2, dataset[3])
    my_device = ""

    try:
        if torch.backends.mps.is_built():
            my_device = "mps"
    except AttributeError:
        if torch.cuda.is_available():
            my_device = "cuda:0"
        else:
            my_device = "cpu"

    fl.client.start_numpy_client(server_address="10.1.2.102:8080",
                                 client=CifarClient(cid=random.randint(1, 10),
                                                    trainset=trainset, testset=testset, device=my_device))


if __name__ == "__main__":
    main()