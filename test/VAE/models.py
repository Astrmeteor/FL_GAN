import torch
import torch.nn as nn


class Flatten(nn.Module):
    """Flattens input by reshaping it into a one-dimensional tensor."""

    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    """Unflattens a tensor converting it to a desired shape."""

    def forward(self, input):
        return input.view(-1, 16, 6, 6)


class Net(nn.Module):
    def __init__(self, h_dim=576, z_dim=10) -> None:
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=6, kernel_size=4, stride=2
            ),  # [batch, 6, 15, 15]
            nn.ReLU(),
            nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=5, stride=2
            ),  # [batch, 16, 6, 6]
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=4, stride=2),
            nn.Tanh(),
        )

    def reparametrize(self, h):
        """Reparametrization layer of VAE."""
        mu, logvar = self.fc1(h), self.fc2(h)
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

    def encode(self, x):
        """Encoder of the VAE."""
        h = self.encoder(x)
        z, mu, logvar = self.reparametrize(h)
        return z, mu, logvar

    def decode(self, z):
        """Decoder of the VAE."""
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z_decode = self.decode(z)
        return z_decode, mu, logvar

### New model


class EncoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, n_neurons_in_middle_layer):
        self.n_neurons_in_middle_layer = n_neurons_in_middle_layer
        super().__init__()
        self.bottle = EncoderModule(color_channels, 32, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(32, 64, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(64, 128, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = EncoderModule(128, 256, stride=pooling_kernels[1], kernel=3, pad=1)

    def forward(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        return out.view(-1, self.n_neurons_in_middle_layer)


class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=stride, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.bn(self.convt(x)))


class Decoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        self.decoder_input_size = decoder_input_size
        super().__init__()
        self.m1 = DecoderModule(256, 128, stride=1)
        self.m2 = DecoderModule(128, 64, stride=pooling_kernels[1])
        self.m3 = DecoderModule(64, 32, stride=pooling_kernels[0])
        self.bottle = DecoderModule(32, color_channels, stride=1, activation="sigmoid")

    def forward(self, x):
        out = x.view(-1, 256, self.decoder_input_size, self.decoder_input_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)


class VAE(nn.Module):
    def __init__(self, dataset):
        self.device = ""
        try:
            if torch.backends.mps.is_built():
                self.device = "mps"
        except AttributeError:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"

        assert dataset in ["mnist", "fashion-mnist", "cifar", "stl"]

        super().__init__()
        # # latent features
        self.n_latent_features = 64

        # resolution
        # mnist, fashion-mnist : 28 -> 14 -> 7
        # cifar : 32 -> 8 -> 4
        # stl : 96 -> 24 -> 6
        if dataset in ["mnist", "fashion-mnist"]:
            pooling_kernel = [2, 2]
            encoder_output_size = 7
        elif dataset == "cifar":
            pooling_kernel = [4, 2]
            encoder_output_size = 4
        elif dataset == "stl":
            pooling_kernel = [4, 4]
            encoder_output_size = 6

        # color channels
        if dataset in ["mnist", "fashion-mnist"]:
            color_channels = 1
        else:
            color_channels = 3

        # # neurons int middle layer
        n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size

        # Encoder
        self.encoder = Encoder(color_channels, pooling_kernel, n_neurons_middle_layer)
        # Middle
        self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc3 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(color_channels, pooling_kernel, encoder_output_size)

        # history
        self.history = {"loss": [], "val_loss": []}


    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def _bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def sampling(self):
        # assume latent features space ~ N(0, 1)
        z = torch.randn(64, self.n_latent_features).to(self.device)
        z = self.fc3(z)
        # decode
        return self.decoder(z)

    def forward(self, x):
        # Encoder
        h = self.encoder(x)
        # Bottle-neck
        z, mu, logvar = self._bottleneck(h)
        # decoder
        z = self.fc3(z)
        d = self.decoder(z)
        return d, mu, logvar