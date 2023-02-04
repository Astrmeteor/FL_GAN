"""
load the weights of models and draw figures
latent space
FID
Sampling and reconstruction
"""
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torchvision
from utils import calculate_fid, load_data
from torch.utils.data import DataLoader
import tqdm
import os
import opacus.validators
from opacus import PrivacyEngine
import warnings

from VAE_Torch.vq_vae_gated_pixelcnn_prior import Improved_VQVAE

warnings.filterwarnings("ignore")

features_out_hook = []
def layer_hook(module, inp, out):
    features_out_hook.append(out.data.cpu().numpy())


def latent_distribution(mu, labels, info):
    e = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(mu.detach().cpu())
    plt.figure()
    plt.scatter(e[:, 0], e[:, 1], c=labels, cmap='tab10')
    plt.colorbar(ticks=np.arange(10), boundaries=np.arange(11)-.5)

    latent_save_path = f"Results/{info['dataset']}/{info['client']}/{info['dp']}/Figures/Latent"
    if not os.path.exists(latent_save_path):
        os.makedirs(latent_save_path)
    latent_save_path += f"/latent_distribution_{info['current_epoch']}.png"
    plt.savefig(latent_save_path, dpi=400)
    plt.close()


def imshow(img: torch.Tensor, savepath):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    # plt.show()
    plt.savefig(savepath, dpi=400)
    plt.close()

def validation(args, test_images, test_labels):

    epochs = args.epochs
    DEVICE = args.device
    DATASET = args.dataset
    CLIENT = args.client
    DP = args.dp
    K = args.num_embeddings
    D = args.embedding_dim

    if args.dataset == "mnist" or args.dataset == "fashion-mnist":
        N, H, W = test_images.data.shape
        C = 1
    else:
        N, C, H, W = test_images.data.shape

    # VQ VAE model initialization
    model = Improved_VQVAE(K=K, D=D, channel=C, device=DEVICE).to(DEVICE)

    # val_images, val_labels = next(iter(testLoader))
    torch_size = torchvision.transforms.Resize([299, 299])

    re_val_images = test_images
    if DATASET == "mnist" or DATASET == "fashion-mnist":
        re_val_images = torch.repeat_interleave(test_images, repeats=3, dim=1)

    images_resize = torch_size(re_val_images).to(DEVICE)

    fid_model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
    fid_model.to(DEVICE)
    fid_model.zero_grad()
    fid_model.eval()
    fid = []

    # Validation and draw related figures
    info = {
        "dataset": DATASET,
        "client": CLIENT,
        "dp": DP
    }

    weights_save_path = f"Results/{info['dataset']}/{info['client']}/{info['dp']}/weights"
    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)


    model.zero_grad()
    model.eval()
    for e in tqdm.tqdm(range(epochs)):

        info["current_epoch"] = e + 1

        vq_vae_PATH = weights_save_path + f"/vqvae/weights_central_{info['current_epoch']}.pt"
        model.load_state_dict(torch.load(vq_vae_PATH, map_location=torch.device(DEVICE)))

        recon_images = model(test_images.to(DEVICE))[0]

        # latent_distribution(recon_mu.reshape(recon_mu.shape[0], -1), test_labels, info)

        # 下面是快速计算的FID、调用Inception V3模型来计算

        if DATASET == "mnist" or DATASET == "fashion-mnist":
            recon_repeat_images = torch.repeat_interleave(recon_images, repeats=3, dim=1)
        else:
            recon_repeat_images = recon_images
        global features_out_hook

        features_out_hook = []
        hook1 = fid_model.dropout.register_forward_hook(layer_hook)
        fid_model.eval()
        fid_model(images_resize)
        images_features = np.array(features_out_hook).squeeze()
        hook1.remove()

        features_out_hook = []
        recon_images_resize = torch_size(recon_repeat_images).to(DEVICE)
        hook2 = fid_model.dropout.register_forward_hook(layer_hook)
        fid_model.eval()
        fid_model(recon_images_resize)
        recon_images_features = np.array(features_out_hook).squeeze()
        hook2.remove()

        fid.append(calculate_fid(images_features, recon_images_features))


    print("Save central FID")

    metrics_save_pth = f"Results/{DATASET}/{CLIENT}/{DP}/metrics"
    if not os.path.exists(metrics_save_pth):
        os.makedirs(metrics_save_pth)
    fid_save_path = metrics_save_pth + f"/values_central_{epochs}.npy"
    np.save(fid_save_path, fid)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Draw figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--grad_sample_mode", type=str, default="hooks")

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar",
        help="Dataset",
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
        "--device",
        type=str,
        default="mps",
        help="default GPU ID for model",
    )

    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs to load",
    )

    parser.add_argument(
        "--secure_rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
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
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )

    parser.add_argument(
        "-D",
        "--embedding_dim",
        type=int,
        default=256,  # 64, 256
        help="Embedding dimention"
    )

    parser.add_argument(
        "-K",
        "--num_embeddings",
        type=int,
        default=512,  # 512, 128
        help="Embedding dimention"
    )

    args = parser.parse_args()

    DATASET = args.dataset
    DP = args.dp
    CLIENT = args.client
    DEVICE = args.device

    _, testset = load_data(DATASET)
    print(DEVICE)
    # model = VAE(DATASET, DEVICE)
    # model = VQVAE(3, 64, 512)

    # model.to(DEVICE)
    testLoader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
    test_images, test_labels = next(iter(testLoader))

    """
    if args.dp == "gaussian":
        model = opacus.validators.ModuleValidator.fix(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                     betas=(args.beta1, args.beta2))

        
        if args.clip_per_layer:
            # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
            n_layers = len(
                [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            )
            max_grad_norm = [
                                args.max_per_sample_grad_norm / np.sqrt(n_layers)
                            ] * n_layers
        else:
            max_grad_norm = args.max_per_sample_grad_norm
        
        max_grad_norm = args.max_per_sample_grad_norm
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
        # clipping = "per_layer" if args.clip_per_layer else "flat"
        model, optimizer, testLoader = privacy_engine.make_private(
            module=model,
            data_loader=testLoader,
            optimizer=optimizer,
            noise_multiplier=args.sigma,
            grad_sample_mode=args.grad_sample_mode,
            max_grad_norm=max_grad_norm
            #clipping=clipping
        )
    """

    with torch.no_grad():
        results = validation(args, test_images, test_labels)