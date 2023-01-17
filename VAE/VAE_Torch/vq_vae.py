import torch.nn as nn
import torch

# L2 Distance squared
l2_dist = lambda x, y: (x - y) ** 2


class LayerNorm(nn.LayerNorm):
    def __init__(self, color_conditioning, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_conditioning = color_conditioning

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        if self.color_conditioning:
            x = x.contiguous().view(*(x_shape[:-1] + (3, -1)))
        x = super().forward(x)
        if self.color_conditioning:
            x = x.view(*x_shape)
        return x.permute(0, 3, 1, 2).contiguous()


class StackLayerNorm(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.h_layer_norm = LayerNorm(False, n_filters)
        self.v_layer_norm = LayerNorm(False, n_filters)

    def forward(self, x):
        vx, hx = x.chunk(2, dim=1)
        vx, hx = self.v_layer_norm(vx), self.h_layer_norm(hx)
        return torch.cat((vx, hx), dim=1)


class GatedConv2d(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, k=7, padding=3):
        super().__init__()

        self.vertical = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=k,
                                  padding=padding, bias=False)
        self.horizontal = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(1, k),
                                    padding=(0, padding), bias=False)
        self.vtoh = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1,
                              bias=False)
        self.htoh = nn.Conv2d(out_channels, out_channels, kernel_size=1,
                              bias=False)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        # No need for special color condition masking here since we get to see everything
        self.vmask[:, :, k // 2 + 1:, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2 + 1:] = 0
        if mask_type == 'A':
            self.hmask[:, :, :, k // 2] = 0

    def down_shift(self, x):
        x = x[:, :, :-1, :]
        pad = nn.ZeroPad2d((0, 0, 1, 0))
        return pad(x)

    def forward(self, x):
        vx, hx = x.chunk(2, dim=1)

        self.vertical.weight.data *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx = self.vertical(vx)
        hx_new = self.horizontal(hx)
        # Allow horizontal stack to see information from vertical stack
        hx_new = hx_new + self.vtoh(self.down_shift(vx))

        # Gates
        vx_1, vx_2 = vx.chunk(2, dim=1)
        vx = torch.tanh(vx_1) * torch.sigmoid(vx_2)

        hx_1, hx_2 = hx_new.chunk(2, dim=1)
        hx_new = torch.tanh(hx_1) * torch.sigmoid(hx_2)
        hx_new = self.htoh(hx_new)
        hx = hx + hx_new

        return torch.cat((vx, hx), dim=1)


class GatedPixelCNN(nn.Module):
    """
    The following Gated PixelCNN is taken from class material given on Piazza
    """

    def __init__(self, K, in_channels=64, n_layers=15, n_filters=256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=K, embedding_dim=in_channels)

        self.in_conv = MaskedConv2d('A', in_channels, n_filters, kernel_size=7, padding=3)
        model = []
        for _ in range(n_layers - 2):
            model.extend([nn.ReLU(), GatedConv2d('B', n_filters, n_filters, 7, padding=3)])
            model.append(StackLayerNorm(n_filters))

        self.out_conv = MaskedConv2d('B', n_filters, K, kernel_size=7, padding=3)
        self.net = nn.Sequential(*model)

    def forward(self, x):
        z = self.embedding(x).permute(0, 3, 1, 2).contiguous()

        out = self.in_conv(z)
        out = self.net(torch.cat((out, out), dim=1)).chunk(2, dim=1)[1]
        out = self.out_conv(out)
        return out

#class MaskedConv2d(nn.Conv2d):
class MaskedConv2d(nn.Module):
    """
    Class extending nn.Conv2d to use masks.
    """

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding=0):
        super(MaskedConv2d, self).__init__()

        self.process = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        #nn.Conv2d.__init__(in_channels, out_channels, kernel_size, padding=padding)
        self.register_buffer('mask', torch.ones(out_channels, in_channels, kernel_size, kernel_size).float())

        # _, depth, height, width = self.weight.size()
        h, w = kernel_size, kernel_size

        if mask_type == 'A':
            self.mask[:, :, h // 2, w // 2:] = 0
            self.mask[:, :, h // 2 + 1:, :] = 0
        else:
            self.mask[:, :, h // 2, w // 2 + 1:] = 0
            self.mask[:, :, h // 2 + 1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return self.process.forward(x)
        # return super().forward(x)


class VectorQuantizer(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.K = K
        self.D = D
        self.codebook = nn.Embedding(num_embeddings=K, embedding_dim=D)
        self.codebook.weight.data.uniform_(-1 / K, 1 / K)

    def forward(self, z_e):
        N, D, H, W = z_e.size()
        z_e = z_e.permute(0, 2, 3, 1).contiguous()  # (N, D, H/4, W/4) --> (N, H/4, W/4, D)
        z_e = z_e.view(-1, self.D)

        weights = self.codebook.weight

        # Sampling nearest embeddings
        distances = l2_dist(z_e[:, None], weights[None, :])
        q = distances.sum(dim=2).min(dim=1)[1]  # Using [1] to get indices instead of values after min-function
        z_q = weights[q]

        # (N, H/4, W/4, D) -> (N, D, H/4, W/4)
        z_q = z_q.view(N, H, W, D)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        q = q.long().view(N, H, W)

        # Class vector q, and code vector z_q
        return q, z_q


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x) + x


class Encoder(nn.Module):
    def __init__(self, D=256, in_channel=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=D, kernel_size=4, stride=2, padding=1),  # 16 x 16
            nn.BatchNorm2d(D), nn.ReLU(),
            nn.Conv2d(in_channels=D, out_channels=D, kernel_size=4, stride=2, padding=1),  # 8 x 8
            ResidualBlock(D),
            ResidualBlock(D)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, D=256, out_channel=3):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(D),
            ResidualBlock(D),
            nn.BatchNorm2d(D), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=D, out_channels=D, kernel_size=4, stride=2, padding=1),  # 16 x 16
            nn.BatchNorm2d(D), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=D, out_channels=out_channel, kernel_size=4, stride=2, padding=1),  # 32 x 32
        )

    def forward(self, x):
        return self.net(x)


class VQVAE(nn.Module):
    def __init__(self, K=128, D=256, channel=3):
        super().__init__()
        #self.K = K
        #self.D = D

        self.codebook = VectorQuantizer(K=K, D=D)
        self.encoder = Encoder(D=D, in_channel=channel)
        self.decoder = Decoder(D=D, out_channel=channel)
        # self.pixelcnn_prior = GatedPixelCNN(K=K).to(device)
        # self.pixelcnn_loss_fct = nn.CrossEntropyLoss()

    def forward(self, x):
        z_e = self.encoder(x)

        q, z_q = self.codebook(z_e)

        # if prior_only: return q, self.pixelcnn_prior(q)

        # z_e_altered = (z_q - z_e).detach() + z_e

        # x_reconstructed = self.decoder(z_e_altered)
        x_reconstructed = self.decoder(z_e)

        return x_reconstructed, z_e, z_q

    def get_encode(self, x):
        z_e = self.encoder(x)
        q, z_q = self.codebook(z_e)
        return q

    """
    def get_pixelcnn_prior_loss(self, x, output):
        q, logit_probs = output
        return self.pixelcnn_loss_fct(logit_probs, q)

    def get_vae_loss(self, x, output):
        N, C, H, W = x.shape
        x_reconstructed = output
        z_e = self.z_e
        z_q = self.z_q

        reconstruction_loss = l2_dist(x, x_reconstructed).sum() / (N * H * W * C)
        vq_loss = l2_dist(z_e.detach(), z_q).sum() / (N * H * W * C)
        commitment_loss = l2_dist(z_e, z_q.detach()).sum() / (N * H * W * C)

        return reconstruction_loss + vq_loss + commitment_loss
    """
