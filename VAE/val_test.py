from models import VAE
from utils import Laplace_mech, Gaussian_mech, load_data, calculate_fid
import torch
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision
import time

my_device = ""

try:
    if torch.backends.mps.is_built():
        my_device = "mps"
except AttributeError:
    if torch.cuda.is_available():
        my_device = "cuda:0"
    else:
        my_device = "cpu"

DEVICE = torch.device(my_device)


def train(net, trainloader, testloader, epochs, dp: str = "", device: str = "cpu"):
    """Train the network on the training set."""
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    net.train()
    LOSS = 0.0
    fid = []
    start_time = time.time()

    for e in tqdm.tqdm(range(epochs)):
        loop = tqdm.tqdm((trainloader), total=len(trainloader), leave=False)
        #for j, (images, labels) in enumerate(trainloader):
        for images, labels in loop:

            optimizer.zero_grad()

            if dp == "laplace":
                images = Laplace_mech(images).float()
            if dp == "gaussian":
                images = Gaussian_mech(images).float()

            images = images.to(device)
            recon_images, mu, logvar = net(images)

            # recon_loss = F.mse_loss(recon_images, images)
            recon_loss = F.binary_cross_entropy(recon_images, images, reduction='sum')
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + * kld_loss
            LOSS += loss
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{e}/{epochs}]")
            loop.set_postfix(loss=loss.item())
        # fid_e = calculate_fid(images[:16].cpu().detach().numpy().reshape(16, -1),
        #                      recon_images[:16].cpu().detach().numpy().reshape(16, -1))

        val_images, val_labels = next(iter(testloader))

        recon_images, _, _ = net(val_images.to(device))

        predictions = recon_images[:16]

        # 下面是快速计算的FID、调用Inception V3模型来计算
        torch_size = torchvision.transforms.Resize([299, 299])

        images_resize = torch_size(recon_images).to(device)

        fid_model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
        fid_model.to(device)

        global features_out_hook

        features_out_hook = []
        hook1 = fid_model.dropout.register_forward_hook(layer_hook)
        fid_model.eval()
        fid_model(images_resize)
        images_features = np.array(features_out_hook).squeeze()
        hook1.remove()

        features_out_hook = []
        recon_images_resize = torch_size(recon_images).to(device)
        hook2 = fid_model.dropout.register_forward_hook(layer_hook)
        fid_model.eval()
        fid_model(recon_images_resize)
        recon_images_features = np.array(features_out_hook).squeeze()
        hook2.remove()

        del fid_model

        fid.append(calculate_fid(images_features, recon_images_features))
        fig = plt.figure(figsize=(3, 3))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            img = np.transpose(predictions[i, :, :, ].cpu().detach().numpy(), axes=[1, 2, 0])
            plt.imshow(img)
            plt.axis('off')
            # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig('G_data/image_at_epoch_{:03d}_G.png'.format(e))
        plt.close()

        grand_truth_images = val_images[:16]
        fig = plt.figure(figsize=(3, 3))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            img = np.transpose(grand_truth_images[i, :, :, ].cpu().detach().numpy(), axes=[1, 2, 0])
            plt.imshow(img)
            plt.axis('off')
            # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig('G_data/image_at_epoch_{:03d}.png'.format(e))
        plt.close()

        label_names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
        val_labels = [label_names[i] for i in np.array(val_labels)]

        np.savetxt("G_data/Labels/label_at_epoch_{:03d}.txt".format(e), val_labels, fmt="%s")

    end_time = time.time()
    train_time = end_time - start_time

    values = {
        "FID": fid,
        "Train_time": train_time
    }

    print("Save central values ")
    np.save(f"Results/values_central_{epochs}.npy", values)

    print("Save central weights")
    torch.save(net.state_dict(), f"Results/weights_central_{epochs}.pt")

features_out_hook = []
def layer_hook(module, inp, out):
    features_out_hook.append(out.data.cpu().numpy())


def imshow(img):
    img = img / 2 + 0.5 # 逆正则化
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    dp = ""
    dataset = ["mnist", "fashion-mnist", "cifar", "stl"]
    trainset, testset = load_data(dataset[3])
    print(DEVICE)
    net = VAE(dataset[3]).to(DEVICE)

    trainLoader = DataLoader(trainset, batch_size=64, shuffle=True)
    testLoader = DataLoader(testset, batch_size=64, shuffle=True)

    epoch = 100
    results = train(net, trainLoader, testLoader, epoch, dp, DEVICE)