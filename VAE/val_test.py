import argparse

import opacus.validators

# from models import VAE
from Models.vq_vae import VQVAE
from Models.gated_pixel_cnn import GatedPixelCNN
from utils import load_data
import torch
from torch.utils.data import DataLoader
import tqdm
from opacus import PrivacyEngine
import numpy as np
import os


def train(args, net, trainloader):
    """Train the network on the training set."""
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    DATASET = args.dataset
    DP = args.dp
    CLIENT = args.client

    device = args.device
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    # gamma = 0.5
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    LOSS = []
    CNN_LOSS = []
    EPSION = []

    if args.dp == "gaussian":
        net = opacus.validators.ModuleValidator.fix(net)
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
            data_loader=trainloader,
            noise_multiplier=args.sigma,
            max_grad_norm=max_grad_norm,
            clipping=clipping,
            grad_sample_mode=args.grad_sample_mode,
        )

    net.train()

    GatedPixelCNN_model = GatedPixelCNN(1, args.num_embeddings, 1).to(device)
    optimizer_cnn = torch.optim.Adam(GatedPixelCNN_model.parameters(), lr=5e-4)

    for e in tqdm.tqdm(range(args.epochs)):
        loop = tqdm.tqdm((trainloader), total=len(trainloader), leave=False)
        temp_loss = []
        temp_cnn_loss = []
        if args.lr_schedule == "cos":
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * e / (args.epochs + 1)))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        for images, labels in loop:
            # VQVAE training
            optimizer.zero_grad()

            images = images.to(device)
            #recon_images, mu, logvar = net(images)
            recon_images, encoding_inds, latent_shape, loss = net(images)

            loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1, norm_type=2)
            #torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            temp_loss.append(loss["loss"].item())

            # GatedPixelCNN traning
            B, _, H, W = latent_shape
            cnn_input = encoding_inds.view(B, 1, H, W)
            cnn_output = GatedPixelCNN_model(cnn_input.float())
            cnn_output = cnn_output.permute(0, 2, 3, 1).contiguous()
            cnn_output = cnn_output.view(-1, 1)

            cnn_loss = torch.nn.functional.binary_cross_entropy(cnn_output, encoding_inds.float())
            optimizer_cnn.zero_grad()
            cnn_loss.backward()
            optimizer_cnn.step()

            temp_cnn_loss.append(cnn_loss.item())

            loop.set_description(f"Epoch [{e}/{args.epochs}]")
            loop.set_postfix(loss=loss["loss"].item(), CNN_loss=cnn_loss)

        if args.dp == "gaussian":
            epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
            print(
                f"Train Epoch: {e} \t"
                f"Loss: {np.mean(temp_loss):.6f} CNN_Loss: {np.mean(temp_cnn_loss)}"
                f"(ε = {epsilon:.2f}, δ = {args.delta})"
            )
            EPSION.append(epsilon.item())
        else:
            print(f"Train Epoch: {e} \t Loss: {np.mean(temp_loss):.6f} CNN_Loss: {np.mean(temp_cnn_loss)}")
        LOSS.append(np.mean(temp_loss).item())

        print("Model has been saved.")
        # Save each epoch's model
        weight_save_pth = f"Results/{DATASET}/{CLIENT}/{DP}/weights/vqvae"
        cnn_weight_save_pth = f"Results/{DATASET}/{CLIENT}/{DP}/weights/gatedpixelcnn"
        if not os.path.exists(weight_save_pth):
            os.makedirs(weight_save_pth)
        weight_save_pth += f"/weights_central_{e}.pt"
        torch.save(net.state_dict(), weight_save_pth)

        if not os.path.exists(cnn_weight_save_pth):
            os.makedirs(cnn_weight_save_pth)
        weight_save_pth += f"/cnn_weights_central_{e}.pt"
        torch.save(net.state_dict(), cnn_weight_save_pth)

    metrics_save_pth = f"Results/{DATASET}/{CLIENT}/{DP}/metrics"
    if not os.path.exists(metrics_save_pth):
        os.makedirs(metrics_save_pth)

    # Save loss
    loss_save_pth = metrics_save_pth + f"/loss_{args.epochs}.npy"
    np.save(loss_save_pth, LOSS)

    # Save Cnn loss
    cnn_loss_save_pth = metrics_save_pth + f"/cnn_loss_{args.epochs}.npy"
    np.save(cnn_loss_save_pth, CNN_LOSS)

    # Save epsilon
    if args.dp == "gaussian":
        epsilon_save_path = metrics_save_pth + f"/epsilon_{args.epochs}.npy"
        np.save(epsilon_save_path, EPSION)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(
        description="Opacus Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--grad_sample_mode", type=str, default="hooks")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=64,
        metavar="B",
        help="Batch size",
    )

    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing",
    )

    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate",
    )

    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )

    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 for adam. default=0.999"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-05,
        metavar="WD",
        help="weight decay",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )

    parser.add_argument(
        "-c",
        "--max_per_sample_grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="default GPU ID for model",
    )

    """
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    """

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="mnist, fashion-mnist, cifar, stl",
    )

    parser.add_argument(
        "--dp",
        type=str,
        default="normal",
        help="Disable privacy training and just train with vanilla type",
    )

    parser.add_argument(
        "--client",
        type=int,
        default=1,
        help="Number of clients, 1 for centralized, 2/3/4/5 for federated learning",
    )

    parser.add_argument(
        "--secure_rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )

    parser.add_argument(
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )

    parser.add_argument(
        "--lr_schedule", type=str, choices=["constant", "cos"], default="cos"
    )

    parser.add_argument(
        "-D",
        "--embedding_dim",
        type=int,
        default=64,
        help="Embedding dimention"
    )

    parser.add_argument(
        "-K",
        "--num_embeddings",
        type=int,
        default=512,
        help="Embedding dimention"
    )


    args = parser.parse_args()

    DATASET = args.dataset
    DP = args.dp
    CLIENT = args.client
    trainset, testset = load_data(DATASET)
    # args.device = "cpu"

    print(f"Dataset:{DATASET}")
    print(f"Client: {CLIENT}")
    print(f"Device:{args.device}")
    print(f"Differential Privacy: {DP}")

    # net = VAE(DATASET, args.device).to(args.device)

    trainLoader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    # VAVAE(shape, D, K)
    net = VQVAE(next(iter(trainLoader))[0].shape[1], args.embedding_dim, args.num_embeddings).to(args.device)
    results = train(args, net, trainLoader)