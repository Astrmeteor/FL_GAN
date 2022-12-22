import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torchvision
from utils import calculate_fid, load_data
from models import VAE
from torch.utils.data import DataLoader
import tqdm
import os
import opacus.validators
from opacus import PrivacyEngine
import sys, warnings

warnings.filterwarnings("ignore")

features_out_hook = []
def layer_hook(module, inp, out):
    features_out_hook.append(out.data.cpu().numpy())


def get_labels(dataset_name: str = "stl"):
    if dataset_name == "stl":
        label_names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    elif dataset_name == "cifar":
        label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif dataset_name == "fashion-mnist":
        label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                       "Ankle boot"]
    elif dataset_name == "mnist":
        label_names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    return label_names


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


def validation(args, model, val_images, val_labels):

    epochs = args.epochs
    DEVICE = args.device
    DATASET = args.dataset
    CLIENT = args.client
    DP = args.dp

    # val_images, val_labels = next(iter(testLoader))
    torch_size = torchvision.transforms.Resize([299, 299])

    re_val_images = val_images
    if DATASET == "mnist" or DATASET == "fashion-mnist":
        re_val_images = torch.repeat_interleave(val_images, repeats=3, dim=1)

    images_resize = torch_size(re_val_images).to(DEVICE)

    fid_model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
    fid_model.to(DEVICE)
    fid_model.zero_grad()
    fid_model.eval()
    fid = []

    #  Save grand truth images
    grand_truth_images = val_images[:16]
    fig = plt.figure(figsize=(3, 3))

    for i in range(grand_truth_images.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = np.transpose(grand_truth_images[i, :, :, ].cpu().detach().numpy(), axes=[1, 2, 0])
        if DATASET != "mnist" and DATASET != "fashion-mnist":
            plt.imshow(img)
        else:
            plt.imshow(img, cmap="gray")
        plt.axis('off')

    grand_images_save_path = f"Results/{DATASET}/{CLIENT}/{DP}/G_data"
    if not os.path.exists(grand_images_save_path):
        os.makedirs(grand_images_save_path)

    grand_images_save_path += f"/grand_truth_images.png"
    plt.savefig(grand_images_save_path, dpi=400)
    plt.close()

    label_names = get_labels(DATASET)

    val_labels_name = [label_names[i] for i in np.array(val_labels)]

    label_save_path = f"Results/{DATASET}/{CLIENT}/{DP}/Labels"
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)

    label_save_path += f"/validation_labels.txt"
    np.savetxt(label_save_path, val_labels_name, fmt="%s")

    # Validation and draw related figures
    info = {
        "dataset": DATASET,
        "client": CLIENT,
        "dp": DP
    }

    weights_save_path = f"Results/{info['dataset']}/{info['client']}/{info['dp']}/weights"
    reconstruction_save_path = f"Results/{info['dataset']}/{info['client']}/{info['dp']}/G_data/Reconstruction"
    sampling_save_path = f"Results/{info['dataset']}/{info['client']}/{info['dp']}/G_data/Sampling"
    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)
    if not os.path.exists(reconstruction_save_path):
        os.makedirs(reconstruction_save_path)
    if not os.path.exists(sampling_save_path):
        os.makedirs(sampling_save_path)

    model.zero_grad()
    model.eval()
    for e in tqdm.tqdm(range(epochs)):

        info["current_epoch"] = e

        PATH = weights_save_path + f"/weights_central_{info['current_epoch']}.pt"
        model.load_state_dict(torch.load(PATH, map_location=torch.device(DEVICE)))
        # model = torch.nn.DataParallel(model)

        recon_images, recon_mu, _ = model(val_images.to(DEVICE))
        latent_distribution(recon_mu, val_labels, info)

        # 下面是快速计算的FID、调用Inception V3模型来计算

        if DATASET == "mnist" or DATASET == "fashion-mnist":
            recon_images = torch.repeat_interleave(recon_images, repeats=3, dim=1)
        global features_out_hook

        features_out_hook = []
        hook1 = fid_model.dropout.register_forward_hook(layer_hook)
        fid_model.eval()
        fid_model(images_resize)
        images_features = np.array(features_out_hook).squeeze()
        hook1.remove()

        features_out_hook = []
        recon_images_resize = torch_size(recon_images).to(DEVICE)
        hook2 = fid_model.dropout.register_forward_hook(layer_hook)
        fid_model.eval()
        fid_model(recon_images_resize)
        recon_images_features = np.array(features_out_hook).squeeze()
        hook2.remove()

        fid.append(calculate_fid(images_features, recon_images_features))

        # Draw reconstruction images
        predictions = recon_images[:16]
        fig = plt.figure(figsize=(3, 3))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            img = np.transpose(predictions[i, :, :, ].cpu().detach().numpy(), axes=[1, 2, 0])
            if DATASET != "mnist" and DATASET != "fashion-mnist":
                plt.imshow(img)
            else:
                plt.imshow(img, cmap="gray")
            plt.axis('off')

        recon_images_save_path = reconstruction_save_path + f"/reconstruction_images_at_epoch_{info['current_epoch']:03d}_G.png"
        plt.savefig(recon_images_save_path, dpi=400)
        plt.close()

        # Draw sampling images
        randoms = torch.Tensor(np.random.normal(0, 1, (64, 128)))
        generated_images = model.decoder(model.fc3(randoms.to(DEVICE)))
        fig = plt.figure(figsize=(8, 8))
        for i in range(generated_images.shape[0]):
            plt.subplot(8, 8, i + 1)
            img = np.transpose(generated_images[i, :, :, ].cpu().detach().numpy(), axes=[1, 2, 0])
            if DATASET != "mnist" and DATASET != "fashion-mnist":
                plt.imshow(img)
            else:
                plt.imshow(img, cmap="gray")
            plt.axis('off')
            # tight_layout minimizes the overlap between 2 sub-plots

        sample_images_save_path = sampling_save_path + f"/sampling_images_at_epoch_{info['current_epoch']:03d}.png"
        plt.savefig(sample_images_save_path, dpi=400)
        plt.close()
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
        default="mnist",
        help="Dataset",
    )

    parser.add_argument(
        "--dp",
        type=str,
        default="gaussian",
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
        default="cpu",
        help="default GPU ID for model",
    )

    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=64,
        metavar="TB",
        help="input batch size for testing",
    )

    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=200,
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
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )

    args = parser.parse_args()

    DATASET = args.dataset
    DP = args.dp
    CLIENT = args.client
    DEVICE = args.device

    _, testset = load_data(DATASET)
    print(DEVICE)
    model = VAE(DATASET, DEVICE)

    model.to(DEVICE)
    testLoader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    val_images, val_labels = next(iter(testLoader))

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

        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
        clipping = "per_layer" if args.clip_per_layer else "flat"
        model, optimizer, testLoader = privacy_engine.make_private(
            module=model,
            data_loader=testLoader,
            optimizer=optimizer,
            noise_multiplier=args.sigma,
            grad_sample_mode=args.grad_sample_mode,
            max_grad_norm=max_grad_norm,
            clipping=clipping
        )

    results = validation(args, model, val_images, val_labels)