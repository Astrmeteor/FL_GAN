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
    plt.savefig('Results/{}/{}/{}/Figures/Latent/latent_distribution_{}.png'.format(
        info["dataset"], info["client"], info["dp"], info["current_epoch"]), dpi=400)
    plt.close()


def validation(args, model, testLoader):

    epochs = args.epochs
    DEVICE = args.device
    DATASET = args.dataset
    CLIENT = args.client
    DP = args.dp

    val_images, val_labels = next(iter(testLoader))
    torch_size = torchvision.transforms.Resize([299, 299])

    re_val_images = val_images
    if DATASET == "mnist" or DATASET == "fashion-mnist":
        re_val_images = torch.repeat_interleave(val_images, repeats=3, dim=1)

    images_resize = torch_size(re_val_images).to(DEVICE)

    fid_model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
    fid_model.to(DEVICE)

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
    plt.savefig('Results/{}/{}/{}/G_data/grand_truth_images.png'.format(DATASET, CLIENT, DP))
    plt.close()

    label_names = get_labels(DATASET)

    val_labels_name = [label_names[i] for i in np.array(val_labels)]

    np.savetxt("Results/{}/{}/{}/Labels/validation_labels.txt".format(DATASET, CLIENT, DP), val_labels_name, fmt="%s")

    # Validation and draw related figures
    for e in tqdm.tqdm(range(epochs)):
        info = {
            "dataset": DATASET,
            "client": CLIENT,
            "dp": DP,
            "current_epoch": e
        }
        PATH = "Results/{}/{}/{}/weights/weights_central_{}.pt".format(info["dataset"], info["client"], info["dp"], info["current_epoch"])
        model.load_state_dict(torch.load(PATH))
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
        plt.savefig('Results/{}/{}/{}/G_data/Reconstruction/reconstruction_images_at_epoch_{:03d}_G.png'.format(DATASET, CLIENT, DP, e))
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
        plt.savefig('Results/{}/{}/{}/G_data/Sampling/sampling_images_at_epoch_{:03d}.png'.format(DATASET, CLIENT, DP, e))
        plt.close()
    print("Save central FID")
    np.save(f"Results/{DATASET}/{CLIENT}/{DP}/metrics/values_central_{epochs}.npy", fid)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Draw figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
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
        default="cuda",
        help="default GPU ID for model",
    )

    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=500,
        metavar="TB",
        help="input batch size for testing",
    )

    args = parser.parse_args()

    try:
        if torch.backends.mps.is_built():
            args.device = "mps"
        else:
            if torch.cuda.is_available():
                args.device = "cuda:0"
            else:
                args.device = "cpu"
    except AttributeError:
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    DATASET = args.dataset
    DP = args.dp
    CLIENT = args.client
    DEVICE = args.device

    _, testset = load_data(DATASET)
    print(DEVICE)
    model = VAE(DATASET).to(DEVICE)
    testLoader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    results = validation(args, model, testLoader)