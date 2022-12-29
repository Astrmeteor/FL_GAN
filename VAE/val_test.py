import argparse

from VAE_Torch.vq_vae_gated_pixelcnn_prior import train_vq_vae_with_gated_pixelcnn_prior
from utils import load_data
import numpy as np
import os


def train(args, trainset, testset):
    """Train the network on the training set."""

    DATASET = args.dataset
    DP = args.dp
    CLIENT = args.client
    if DP == "gaussian":
        vq_vae_train_losses, vq_vae_test_losses, pixel_cnn_train_losses, pixel_cnn_test_losses = train_vq_vae_with_gated_pixelcnn_prior(args, trainset, testset)
    else:
        vq_vae_train_losses, vq_vae_test_losses, pixel_cnn_train_losses, pixel_cnn_test_losses = train_vq_vae_with_gated_pixelcnn_prior(args, trainset, testset)

    metrics_save_pth = f"Results/{DATASET}/{CLIENT}/{DP}/metrics"
    if not os.path.exists(metrics_save_pth):
        os.makedirs(metrics_save_pth)

    # Save VQVAE Train Loss
    vqvqe_train_loss_save_pth = metrics_save_pth + f"/vqvqe_train_loss_{args.epochs}.npy"
    np.save(vqvqe_train_loss_save_pth, vq_vae_train_losses)

    # Save VQVAE Test Loss
    vqvae_test_loss_save_pth = metrics_save_pth + f"/vqvae_test_loss_{args.epochs}.npy"
    np.save(vqvae_test_loss_save_pth, vq_vae_test_losses)

    # Save PixelCNN Train Loss
    cnn_train_loss_save_pth = metrics_save_pth + f"/cnn_train_loss_{args.epochs}.npy"
    np.save(cnn_train_loss_save_pth, pixel_cnn_train_losses)

    # Save PixelCNN Test Loss
    cnn_test_loss_save_pth = metrics_save_pth + f"/cnn_test_loss_{args.epochs}.npy"
    np.save(cnn_test_loss_save_pth, pixel_cnn_test_losses)

    # Save epsilon
    if args.dp == "gaussian":
        epsilon_save_path = metrics_save_pth + f"/epsilon_{args.epochs}.npy"
        np.save(epsilon_save_path, epsilon)


def args_function():
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
        default=1,
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
        default=256, # 64, 256
        help="Embedding dimention"
    )

    parser.add_argument(
        "-K",
        "--num_embeddings",
        type=int,
        default=512, #512, 128
        help="Embedding dimention"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = args_function()
    DATASET = args.dataset
    DP = args.dp
    CLIENT = args.client

    trainset, testset = load_data(DATASET)

    print(f"Dataset:{DATASET}")
    print(f"Client: {CLIENT}")
    print(f"Device:{args.device}")
    print(f"Differential Privacy: {DP}")

    results = train(args, trainset, testset)