import time
import torch
import torch.nn as nn
import numpy as np

from .utils import get_batched_loss
from .vq_vae import MaskedConv2d, VQVAE
import torch.nn.functional as F
import tqdm
from opacus import PrivacyEngine
import opacus
import os


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


class Improved_VQVAE(VQVAE):
    """
    VQ-VAE using Gated PixelCNN
    """

    def __init__(self, K=128, D=256, channel=3):
        super().__init__(K, D, channel)
        self.pixelcnn_prior = GatedPixelCNN(K=K)


def dequantize(x, dequantize=True, reverse=False, alpha=.1):
    with torch.no_grad():
        if reverse:
            return torch.ceil_(torch.sigmoid(x) * 255)

        if dequantize:
            x += torch.zeros_like(x).uniform_(0, 1)

        p = alpha / 2 + (1 - alpha) * x / 256
        return torch.log(p) - torch.log(1 - p)


def train_vq_vae_with_gated_pixelcnn_prior(args, train_set, validation_set, test_set):
    """
    args: parameters for code
    train_set
    validation_set
    test_set

    Returns
    - a (# of training iterations,) numpy array of VQ-VAE train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of VQ-VAE train losses evaluated once at initialization and after each epoch
    - a (# of training iterations,) numpy array of PixelCNN prior train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of PixelCNN prior train losses evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples (an equal number from each class) with values in {0, ... 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
        FROM THE TEST SET with values in [0, 255]
    """
    start_time = time.time()
    if args.dataset == "mnist" or args.dataset == "fashion-mnist":
        N, H, W = train_set.dataset.data.shape
        C = 1
    else:
        N, H, W, C = train_set.dataset.data.shape

    batch_size = args.batch_size
    dataset_params = {
        'batch_size': batch_size,
        'shuffle': False #True
    }

    print("[INFO] Creating model and data loaders")
    train_loader = torch.utils.data.DataLoader(train_set, **dataset_params)
    validation_loader = torch.utils.data.DataLoader(validation_set, **dataset_params)
    test_loader = torch.utils.data.DataLoader(test_set, **dataset_params)

    # Model
    n_epochs_vae = args.epochs
    n_epochs_cnn = args.epochs
    lr = args.lr
    K = args.num_embeddings
    D = args.embedding_dim
    # vq_vae = Improved_VQVAE(K=K, D=D).cuda()

    vq_vae = Improved_VQVAE(K=K, D=D, channel=C).to(device=args.device)

    optimizer = torch.optim.Adam(vq_vae.parameters(), lr=lr)

    if args.dp == "gaussian":
        net = opacus.validators.ModuleValidator.fix(vq_vae)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                     betas=(args.beta1, args.beta2))
        if args.clip_per_layer:
            # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
            n_layers = len(
                [(n, p) for n, p in net.named_parameters() if p.requires_grad]
            )
            max_grad_norm = [
                                args.max_per_sample_grad_norm / np.sqrt(n_layers)
                            ] * n_layers
        else:
            max_grad_norm = args.max_per_sample_grad_norm

        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
        clipping = "per_layer" if args.clip_per_layer else "flat"
        net, optimizer, train_loader = privacy_engine.make_private(
            module=net,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=max_grad_norm,
            clipping=clipping,
            grad_sample_mode=args.grad_sample_mode,
        )

    # Training
    def train(args, model, no_epochs, prior_only=False):
        """
        Trains model and returns training and test losses
        """

        DATASET = args.dataset
        DP = args.dp
        CLIENT = args.client

        device = args.device
        model_name = "VQ-VAE" if not prior_only else "Gated Pixel-CNN"
        print(f"[INFO] Training {model_name}")
        model.to(device)
        loss_fct = model.get_vae_loss if not prior_only else model.get_pixelcnn_prior_loss

        train_losses = []
        test_losses = []
        validate_losses = []
        # Initialization loss for test
        # test_losses = [get_batched_loss(args, test_loader, model, loss_fct, prior_only, loss_triples=False)]
        if args.dp == "gaussian":
            EPSILON = []

        for epoch in tqdm.tqdm(range(no_epochs)):
            epoch_start = time.time()
            loop = tqdm.tqdm((train_loader), total=len(train_loader), leave=False)
            for images, labels in loop:
                if loop.last_print_n == 1:
                    break

                optimizer.zero_grad()
                batch = images.to(device)
                output = model(batch, prior_only)
                loss = loss_fct(batch, output)
                loss.backward()
                optimizer.step()
                # batch = batch.cpu()

                train_losses.append(loss.cpu().item())

                loop.set_description(f"Epoch [{epoch}/{args.epochs}]")
                loop.set_postfix(loss=loss.item())

            validate_loss = get_batched_loss(args, validation_loader, vq_vae, loss_fct, prior_only, loss_triples=False)
            validate_losses.append(validate_loss)

            test_loss = get_batched_loss(args, test_loader, vq_vae, loss_fct, prior_only, loss_triples=False)
            test_losses.append(test_loss)

            print(
                f"[{100*(epoch+1)/no_epochs:.2f}%] Epoch {epoch + 1} - "
                f"Train loss: {np.mean(train_losses):.2f} - "
                f"Validate loss: {validate_loss:.2f} - "
                f"Test loss: {test_loss:.2f} - "
                f"Time elapsed: {time.time() - epoch_start:.2f}", end=""
            )
            if args.dp == "gaussian":
                epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
                EPSILON.append(epsilon.item())
                print(f"(ε = {epsilon:.2f}, δ = {args.delta})")
            else:
                print("\n")

            # Save each epoch's model
            weight_save_pth = f"Results/{DATASET}/{CLIENT}/{DP}/weights/vqvae"

            if not os.path.exists(weight_save_pth):
                os.makedirs(weight_save_pth)
            weight_save_pth += f"/weights_central_{epoch + 1}.pt"
            torch.save(model.state_dict(), weight_save_pth)

            if prior_only:
                cnn_weight_save_pth = f"Results/{DATASET}/{CLIENT}/{DP}/weights/gatedpixelcnn"
                if not os.path.exists(cnn_weight_save_pth):
                    os.makedirs(cnn_weight_save_pth)
                weight_save_pth = cnn_weight_save_pth + f"/cnn_weights_after_trained_{epoch + 1}.pt"
                torch.save(model.state_dict(), weight_save_pth)
        if not prior_only:
            metrics = {
                "vq_vae_train_losses": np.array(train_losses),
                "vq_vae_validate_losses": np.array(validate_losses),
                "vq_vae_test_losses": np.array(test_losses)
            }
        else:
            metrics = {
                "pixcel_cnn_train_losses": np.array(train_losses),
                "pixcel_cnn_validate_losses": np.array(validate_losses),
                "pixcel_cnn_test_losses": np.array(test_losses)
            }

        if DP == "gaussian":
            metrics["epsilon"] = np.array(EPSILON)
        return metrics

    vq_vqe_metrics = train(args, vq_vae, n_epochs_vae)
    pixel_cnn_metrics = train(args, vq_vae, n_epochs_cnn, prior_only=True)

    """
    # Utility methods
    def sample(no_samples=100, my_device="cpu"):
        shape = (no_samples, H // 4, W // 4)
        q_samples = torch.zeros(size=shape).long().to(my_device)

        for i in range(H // 4):
            for j in range(W // 4):
                out = vq_vae.pixelcnn_prior(q_samples)
                proba = F.softmax(out, dim=1)
                q_samples[:, i, j] = torch.multinomial(proba[:, :, i, j], 1).squeeze().float()

        z_q_samples = vq_vae.codebook.codebook.weight[q_samples.view(-1, D)].float()

        # Shape (N, W, H, D) -> (N, D, W, H)
        z_q_samples = z_q_samples.view(shape + (D,))
        z_q_samples = z_q_samples.permute(0, 3, 1, 2).contiguous()

        x_samples = vq_vae.decoder(z_q_samples)
        # samples = dequantize(x_samples, reverse=True).detach().cpu().numpy()
        samples = x_samples.detach().cpu().numpy()
        return np.transpose(samples, [0, 2, 3, 1])  # Get it in (N, H, W, C)

    def reconstruction_pairs():
        '''
        Creating reconstruction pairs (x, x') where x is the original image and x' is the decoder-output
        '''
        # x_original = test_data.data[:no_reconstructions]  # .float().cuda()
        x_original, _ = next(iter(test_loader))
        # x_dequantized = dequantize(x_original, dequantize=False)
        x_original = x_original.to(args.device)
        x_reconstructed = vq_vae(x_original)[0]
        # x_reconstructed = dequantize(x_reconstructed.float(), reverse=True)

        pairs = torch.zeros_like(torch.cat((x_original, x_reconstructed), dim=0)).detach().cpu().numpy()
        pairs[::2] = x_original.detach().cpu().numpy()
        pairs[1::2] = x_reconstructed.detach().cpu().numpy()

        pairs = np.clip(pairs, 0, 255)
        return np.transpose(pairs, [0, 2, 3, 1])  # Get it in (N, H, W, C)
    

    torch.cuda.empty_cache()
    vq_vae.eval()
    with torch.no_grad():
        print("[INFO] Sampling images")
        samples = sample(no_samples=100, my_device=args.device)

        print("[INFO] Creating reconstructing pairs")
        pairs = reconstruction_pairs()
    """
    print(f"[DONE] Time elapsed: {time.time() - start_time:.2f} s")

    # print("Samples", samples.shape)
    # print("Pairs", pairs.shape)

    return {**vq_vqe_metrics, **pixel_cnn_metrics}
