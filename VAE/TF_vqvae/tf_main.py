import time

import numpy as np

from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_vectorized import VectorizedDPAdam
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from keras import layers

from tf_model import get_vqvae, get_pixel_cnn, VQVAE
from tf_utils import load_data, get_labels, show_batch, show_latent, show_sampling, get_fid_score, get_inception_score, get_psnr

import argparse
import os
import pandas as pd
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore') #, category=DeprecationWarning)
# warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
# warnings.filterwarnings("default")
# parser

parser = argparse.ArgumentParser(
        description="Tensorflow Differential Pirvacy for VQ-VAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

parser.add_argument("--dpsgd", type=bool, default=False,
                    help="If True, train with DP-SGD. If False, train with vanilla SGD.")

parser.add_argument("--learning_rate", "-lr", type=float, default=0.01,
                    help="Learning rate for training")

parser.add_argument("--l2_norm_clip", "-clip", type=float, default=1.0,
                    help="Clipping norm")

parser.add_argument("--noise_multiplier", "-nm", type=float, default=1.3,
                    help="Ratio of the standard deviation to the clipping norm")

parser.add_argument("--epochs", "-e", type=int, default=1,
                    help="Number of epochs")

parser.add_argument("--delta", type=float, default=1e-5,
                    help="Bounds the probability of an arbitrary change in model behavior")

parser.add_argument("--batch_size", "-bs", type=int, default=256,
                    help="Number of batch size")

parser.add_argument("--micro_batch", "-mc", type=int, default=200,
                    help="Number of micro batches (must evenly divide batch_size)")

parser.add_argument("--dataset_len", "-len", type=int, default=60000,
                    help="Number of dataset, 600000 for MNIST")

parser.add_argument("--latent_dim", "-D", type=int, default=128,
                    help="Embedding dimension")

parser.add_argument("--num_embeddings", "-K", type=int, default=256,
                    help="Number embedding")

parser.add_argument("--dataset", type=str, default="cifar10",
                    help="Dataset: mnist, fashion-mnist, cifar10, stl")

parser.add_argument("--recon_num", type=int, default=36, help="Number of reconstruction for image, must be even")

parser.add_argument("--latent_num", type=int, default=10, help="Number of latent for image")

parser.add_argument("--sampling_num", type=int, default=10, help="Number of sampling for image")

parser.add_argument("--epsilon", type=float, default=2.0, help="Fixed value of epsilon")
args = parser.parse_args()


"""
## Wrapping up the training loop inside `VQVAETrainer`
"""


class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=64, num_embeddings=128, data_shape=[], **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.data_shape = data_shape
        # self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings, data_shape)
        self.vqvae = VQVAE(self.num_embeddings, self.latent_dim)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    # @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.

            reconstruction_loss = tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if args.dpsgd == True:
            print("\nDifferential Privacy Information")

            eps, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
                args.dataset_len, args.batch_size, args.noise_multiplier, epoch+1, args.delta)
            logs["epsilon"] = eps

        # Save model of each epoch
        checkpoint_path = iwantto_path + f"/{args.dataset}/v1/{'dp' if args.dpsgd else 'normal'}/model"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path += f"/vqvae_cp_{epoch}"

        # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="vqvae_loss", verbose=1, save_best_only=True, mode="min")
        self.model.save_weights(checkpoint_path)


class Pixel_CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if args.dpsgd == True:
            print("\nDifferential Privacy Information")

            # while train vqvae and pixel cnn separately, it is necessary to add previous epoch into pixel training
            # Because both utilize the same optimizer
            eps, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
                args.dataset_len, args.batch_size, args.noise_multiplier, epoch+1+args.epochs, args.delta)
            logs["epsilon"] = eps

        # Save model of each epoch
        checkpoint_path = iwantto_path + f"/{args.dataset}/v1/{'dp' if args.dpsgd else 'normal'}/model"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path += f"/pixel_cnn_cp_{epoch}"
        self.model.save_weights(checkpoint_path)


iwantto_path = os.path.dirname(__file__)


def main():
    """
    ## Load dataset
    """

    train_data, train_labels, test_data, test_labels = load_data(args.dataset)
    data_variance = np.var(train_data, dtype=np.float32)

    """
    ## Train the VQ-VAE model
    """
    data_shape = train_data.shape[1:]
    vqvae_trainer = VQVAETrainer(data_variance,
                                 latent_dim=args.latent_dim,
                                 num_embeddings=args.num_embeddings,
                                 data_shape=data_shape)
    print(f"Differential Privacy Switch: {args.dpsgd}")
    if args.dpsgd:
        print("Processing in Differential Privacy")
        args.dataset_len = train_data.shape[0]
        magnitude = len(str(args.dataset_len))
        args.delta = pow(10, -magnitude)

        optimizer = VectorizedDPAdam(
            l2_norm_clip=args.l2_norm_clip,
            noise_multiplier=args.noise_multiplier,
            num_microbatches=args.micro_batch,
            learning_rate=args.learning_rate,
            epsilon=1.0
        )
    else:
        print("Processing in normal")
        # optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate)

    vqvae_trainer.compile(optimizer=optimizer)

    print(f"Start to training with {'DP' if args.dpsgd else 'Normal'} for {args.dataset}")

    train_start = time.time()
    history = vqvae_trainer.fit(
        train_data, epochs=args.epochs, batch_size=args.batch_size, callbacks=[CustomCallback()]
    )
    train_end = time.time() - train_start
    print(f"VQ-VAE training time: {train_end:.2f}s")

    # Save metrics
    vqvae_metric_save_path = iwantto_path + f"/{args.dataset}/v1/{'dp' if args.dpsgd else 'normal'}/metric"
    if not os.path.exists(vqvae_metric_save_path):
        os.makedirs(vqvae_metric_save_path)
    vqvae_metric_save_path += f"/vq_vae_metrics_{args.epochs}.csv"
    vq_vae_metrics = pd.DataFrame(history.history)
    vq_vae_metrics.to_csv(vqvae_metric_save_path, index=False)

    """
    Reconstruction results on the test set
    Save True image and generated image
    This is traditional flow: v1
    """
    reconstruction_path_traditional_flow = iwantto_path + f"/{args.dataset}/v1/{'dp' if args.dpsgd else 'normal'}/Images"
    if not os.path.exists(reconstruction_path_traditional_flow):
        os.makedirs(reconstruction_path_traditional_flow)

    truth_path = reconstruction_path_traditional_flow + f"/grand_truth_images_{args.epochs}.png"
    trained_vqvae_model = vqvae_trainer.vqvae
    # idx = np.random.choice(len(test_data), args.recon_num)
    idx = args.recon_num
    test_images = test_data[:idx]

    # Save grand truth label, if available
    label_save_path = reconstruction_path_traditional_flow + "/grand_truth_label.txt"
    label_names = get_labels(args.dataset)
    val_labels_name = [label_names[i] for i in np.array(test_labels[:idx])]
    np.savetxt(label_save_path, val_labels_name, fmt="%s")
    batch_size = int(pow(args.recon_num, 0.5))
    # reconstruction_images = trained_vqvae_model.predict(tf.convert_to_tensor(test_images))
    reconstruction_images = trained_vqvae_model.predict(test_images)
    reconstruction_save_path = reconstruction_path_traditional_flow + f"/reconstruction_image_{args.epochs}.png"

    # test_x = tf.tile(test_data, [1, 1, 1, 3])
    # recon_x = tf.tile(reconstruction_image, [1, 1, 1, 3])

    # Return the [0, 1] to [0, 255]
    fid = get_fid_score(tf.cast(test_images * 255, tf.uint8), tf.cast(reconstruction_images * 255, tf.uint8))
    print(f"Compare test images with reconstruction images, FID: {fid:.2f}")

    inception_score = get_inception_score(tf.cast(reconstruction_images * 255, tf.uint8))
    print(f"Compute reconstruction images, IS: {inception_score:.2f}")

    psnr = get_psnr(tf.cast(test_images * 255, tf.uint8), tf.cast(reconstruction_images * 255, tf.uint8))
    print(f"Peak Signal-to-Noise Ratio, PSNR: {psnr:.2f}")

    show_batch(test_images, batch_size, truth_path, False if test_images.shape[3] == 3 else True)
    show_batch(reconstruction_images, batch_size, reconstruction_save_path, False if reconstruction_images.shape[3] == 3 else True)

    """
    ## Visualizing the discrete codes
    """

    # encoder = vqvae_trainer.vqvae.get_layer("encoder")
    encoder = vqvae_trainer.vqvae.encode
    # quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")
    quantizer = vqvae_trainer.vqvae.vq_layer

    # encoded_outputs = encoder.predict(test_images)
    encoded_outputs = encoder(test_images)

    # flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    flat_enc_outputs = encoded_outputs.numpy().reshape(-1, encoded_outputs.shape[-1])

    # codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = quantizer(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

    latent_save_path = reconstruction_path_traditional_flow + f"/latent_image_{args.epochs}.png"
    show_latent(test_images[:args.latent_num], codebook_indices[:args.latent_num], reconstruction_images[:args.latent_num],
                latent_save_path, False if test_images.shape[3] == 3 else True)

    """
    ## PixelCNN hyper parameters
    """

    pixelcnn_input_shape = encoded_outputs.shape[1:-1]
    print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")
    pixel_cnn = get_pixel_cnn(pixelcnn_input_shape, args.num_embeddings)

    # pixel_cnn.summary()

    """
    ## Prepare data to train the PixelCNN
    """

    # Generate the codebook indices.
    # encoded_outputs = encoder.predict(train_data)
    encoded_outputs = encoder(train_data)
    flat_enc_outputs = encoded_outputs.numpy().reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)

    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")

    """
    ## PixelCNN training
    """

    pixel_cnn.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    pixel_start = time.time()
    pixel_history = pixel_cnn.fit(
        x=codebook_indices,
        y=codebook_indices,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.1,
        callbacks=[Pixel_CustomCallback()]
    )
    pixel_end = time.time() - pixel_start
    print(f"Pixel training time: {pixel_end:.2f}s")

    # Compute accuracy of autoregressive model
    ar_acc = []
    for _ in range(args.epochs):
        idx = np.random.choice(len(test_data), int(len(test_data)*0.8))
        test_images = test_data[idx]

        encoded_outputs = encoder(test_images)
        flat_enc_outputs = encoded_outputs.numpy().reshape(-1, encoded_outputs.shape[-1])
        codebook_indices = quantizer.get_code_indices(flat_enc_outputs)

        codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
        _, acc = pixel_cnn.evaluate(codebook_indices, codebook_indices, batch_size=args.batch_size, verbose=0)
        ar_acc.append(acc)

    # Save Pixel CNN metrics
    pixel_cnn_save_path = iwantto_path + f"/{args.dataset}/v1/{'dp' if args.dpsgd else 'normal'}/metric"
    if not os.path.exists(pixel_cnn_save_path):
        os.makedirs(pixel_cnn_save_path)
    pixel_cnn_metric_save_path = pixel_cnn_save_path + f"/pixel_cnn_metrics_{args.epochs}.csv"
    pixel_cnn_metrics = pd.DataFrame(pixel_history.history)
    pixel_cnn_metrics.to_csv(pixel_cnn_metric_save_path, index=False)

    pixel_cnn_acc_save_path = pixel_cnn_save_path + f"/pixel_cnn_acc_{args.epochs}.csv"
    pd.DataFrame(ar_acc).to_csv(pixel_cnn_acc_save_path, index=False)

    """
    ## Codebook sampling
    """

    # Create a mini sampler model.
    inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
    outputs = pixel_cnn(inputs, training=False)
    categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
    outputs = categorical_layer(outputs)
    sampler = keras.Model(inputs, outputs)

    # Create an empty array of priors.
    batch = args.sampling_num
    priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
    batch, rows, cols = priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in range(rows):
        for col in range(cols):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            probs = sampler.predict(priors, verbose=0)
            # Use the probabilities to pick pixel values and append the values to the priors.
            priors[:, row, col] = probs[:, row, col]
    print(f"Prior shape: {priors.shape}")

    # Perform an embedding lookup.
    pretrained_embeddings = quantizer.embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"), vqvae_trainer.num_embeddings).numpy()
    quantized = tf.matmul(
        priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True
    )
    quantized = tf.reshape(quantized, (-1, *(encoded_outputs.shape[1:])))

    # Generate novel images.
    decoder = vqvae_trainer.vqvae.get_layer("decoder")
    generated_samples = decoder.predict(quantized)

    sampling_save_path = reconstruction_path_traditional_flow + f"/sampling_image_{args.epochs}.png"
    show_sampling(priors, generated_samples, sampling_save_path, False if generated_samples.shape[3] == 3 else True)

    inception_score = get_inception_score(tf.cast(generated_samples * 255, tf.uint8))
    print(f"Compute sampling images, IS: {inception_score:.2f}")


if __name__ == "__main__":
    main()