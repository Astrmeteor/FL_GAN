import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from VAE_Torch.vq_vae import get_batched_loss, VQVAE
from utils import get_labels
import tqdm
from opacus import PrivacyEngine
from opacus.validators import register_module_validator

import opacus
import os
import matplotlib.pyplot as plt


def dequantize(x, dequantize=True, reverse=False, alpha=.1):
    with torch.no_grad():
        if reverse:
            return torch.ceil_(torch.sigmoid(x) * 255)

        if dequantize:
            x += torch.zeros_like(x).uniform_(0, 1)

        p = alpha / 2 + (1 - alpha) * x / 256
        return torch.log(p) - torch.log(1 - p)


def imshow(img: torch.Tensor, savepath):
    img = img/2 + 0.5
    npimg = img.numpy()
    npimg = np.clip(npimg, 0, 255)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    # plt.show()
    plt.savefig(savepath, dpi=400)
    plt.close()

# L2 Distance squared
l2_dist = lambda x, y: (x - y) ** 2


def get_vae_loss(x, output):
    N, C, H, W = x.shape
    x_reconstructed, z_e, z_q = output

    reconstruction_loss = l2_dist(x, x_reconstructed).sum() / (N * H * W * C)
    vq_loss = l2_dist(z_e.detach(), z_q).sum() / (N * H * W * C)
    commitment_loss = l2_dist(z_e, z_q.detach()).sum() / (N * H * W * C)

    return reconstruction_loss + vq_loss + commitment_loss


def get_pixelcnn_prior_loss(x, output):
    q, logit_probs = output
    return nn.CrossEntropyLoss()(logit_probs, q)


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
    elif args.dataset == "stl":
        N, C, H, W = train_set.dataset.data.shape
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

    # Images for showing reconstruction
    recon_images, recon_labels = next(iter(test_loader))
    recon_images = recon_images[:args.num_reconstruction]
    recon_labels = recon_labels[:args.num_reconstruction]

    # Save grand truth labels
    DATASET = args.dataset
    DP = args.dp
    CLIENT = args.client
    label_names = get_labels(DATASET)
    val_labels_name = [label_names[i] for i in np.array(recon_labels)]
    image_save_path = f"Results/{DATASET}/{CLIENT}/{DP}/G_data"
    label_save_path = f"Results/{DATASET}/{CLIENT}/{DP}/Labels"
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)
    image_save_path += f"/grand_truth_images.png"
    label_save_path += f"/validation_labels.txt"
    np.savetxt(label_save_path, val_labels_name, fmt="%s")
    # Save grand truth images
    imshow(torchvision.utils.make_grid(recon_images, nrow=6), image_save_path)

    # Model
    n_epochs_vae = args.epochs
    n_epochs_cnn = args.epochs
    lr = args.lr
    K = args.num_embeddings
    D = args.embedding_dim
    # vq_vae = Improved_VQVAE(K=K, D=D).cuda()

    vq_vae = VQVAE(K=K, D=D, channel=C, device=args.device).to(device=args.device)

    optimizer = torch.optim.Adam(vq_vae.parameters(), lr=lr)

    if args.dp == "gaussian":
        if not opacus.validators.ModuleValidator.is_valid(vq_vae):
            vq_vae = opacus.validators.ModuleValidator.fix(vq_vae)

        optimizer = torch.optim.Adam(vq_vae.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                     betas=(args.beta1, args.beta2))
        if args.clip_per_layer:
            # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
            n_layers = len(
                [(n, p) for n, p in vq_vae.named_parameters() if p.requires_grad]
            )
            max_grad_norm = [
                                args.max_per_sample_grad_norm / np.sqrt(n_layers)
                            ] * n_layers
        else:
            max_grad_norm = args.max_per_sample_grad_norm

        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
        clipping = "per_layer" if args.clip_per_layer else "flat"

        vq_vae, optimizer, train_loader = privacy_engine.make_private(
            module=vq_vae,
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
        print(f"[INFO] Training {model_name} on Device {device}")
        model.to(device)
        loss_fct = get_vae_loss if not prior_only else get_pixelcnn_prior_loss

        train_losses = []
        test_losses = []
        validate_losses = []
        # Initialization loss for test
        # test_losses = [get_batched_loss(args, test_loader, model, loss_fct, prior_only, loss_triples=False)]
        # why needs function "get_batched_loss" ???
        if args.dp == "gaussian":
            EPSILON = []
        min_loss = math.inf
        for epoch in tqdm.tqdm(range(no_epochs)):
            epoch_start = time.time()
            loop = tqdm.tqdm((train_loader), total=len(train_loader), leave=False)
            print(f"\n[INFO] Training ...")
            for images, labels in loop:
                if loop.last_print_n > 0:
                    break

                optimizer.zero_grad()
                batch = images.to(device)
                output = model(batch, prior_only)
                loss = loss_fct(batch, output)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.cpu().item())

                loop.set_description(f"Epoch [{epoch}/{args.epochs}]")
                loop.set_postfix(loss=loss.item())
            print(f"\n[INFO] Validation ...")
            validate_loss = get_batched_loss(args, validation_loader, vq_vae, loss_fct, prior_only, loss_triples=False)
            validate_losses.append(validate_loss)
            print(f"\n[INFO] Test ...")
            test_loss = get_batched_loss(args, test_loader, vq_vae, loss_fct, prior_only, loss_triples=False)
            test_losses.append(test_loss)

            if not prior_only:
                # Save reconstruction images and labels for each epoch
                reconstruction_save_path = f"Results/{DATASET}/{CLIENT}/{DP}/G_data/Reconstruction"
                if not os.path.exists(reconstruction_save_path):
                    os.makedirs(reconstruction_save_path)
                # recon_image is reconstructed images; recon_images is fixed images for reconstruction
                recon_image = model(recon_images.to(device))[0]
                recon_images_save_path = reconstruction_save_path + f"/reconstruction_images_at_epoch_{epoch+1:03d}_G.png"
                imshow(torchvision.utils.make_grid(recon_image.cpu().detach(), nrow=6), recon_images_save_path)
            else:
                # Save and generate sampling images (synthetic images)
                sampling_save_path = f"Results/{DATASET}/{CLIENT}/{DP}/G_data/Sampling"
                if not os.path.exists(sampling_save_path):
                    os.makedirs(sampling_save_path)

                def sample(no_samples=args.num_sampling):
                    shape = (no_samples, H // 4, W // 4)
                    q_samples = torch.zeros(size=shape).long().to(device)

                    for i in range(H // 4):
                        for j in range(W // 4):
                            out = vq_vae.pixelcnn_prior(q_samples)
                            proba = F.softmax(out, dim=1)
                            q_samples[:, i, j] = torch.multinomial(proba[:, :, i, j], 1).squeeze().float()

                    latents_shape = q_samples.shape
                    encoding_inds = q_samples.view(-1, 1)
                    encoding_one_hot = torch.zeros(encoding_inds.size(0), K, device=device)
                    encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]
                    quantized_latents = torch.matmul(encoding_one_hot, vq_vae.codebook.codebook.weight)  # [BHW, D]
                    quantized_latents = quantized_latents.view(latents_shape + (D,))  # [B x H x W x D]
                    z_q_samples = quantized_latents.permute(0, 3, 1, 2).contiguous()

                    #z_q_samples = vq_vae.codebook.codebook.weight[q_samples.view(-1, D)].float()
                    # Shape (N, W, H, D) -> (N, D, W, H)
                    # z_q_samples = z_q_samples.view(shape + (D,))
                    # z_q_samples = z_q_samples.permute(0, 3, 1, 2).contiguous()

                    x_samples = vq_vae.decoder(z_q_samples)
                    # samples = dequantize(x_samples, reverse=True).detach().cpu().numpy()
                    return x_samples.detach().cpu()
                sampling_images = sample(args.num_sampling)
                sample_images_save_path = sampling_save_path + f"/sampling_images_at_epoch_{epoch+1:03d}.png"
                imshow(torchvision.utils.make_grid(sampling_images, nrow=6), sample_images_save_path)

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

            # Save VQ-VAE model with GatedPixelCNN
            if prior_only:
                # Save minimum test loss of the model when after trained VQ VAE
                if test_loss < min_loss:
                    min_loss = test_loss
                    weight_save_pth = f"Results/{DATASET}/{CLIENT}/{DP}/weights"
                    if not os.path.exists(weight_save_pth):
                        os.makedirs(weight_save_pth)
                    weight_save_pth += f"/weights_central_{args.epochs}.pt"
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

    print(f"[DONE] Time elapsed: {time.time() - start_time:.2f} s")
    return {**vq_vqe_metrics, **pixel_cnn_metrics}
