import numpy as np

from tensorflow import keras
import tensorflow as tf

import tensorflow_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_vectorized import VectorizedDPAdam
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from keras import layers

from tf_model import get_vqvae
from tf_utils import load_data, get_labels, show_batch, show_latent

import argparse
import os
import pandas as pd

# parser

parser = argparse.ArgumentParser(
        description="Tensorflow Differential Pirvacy for VQ-VAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

parser.add_argument("--dpsgd", type=bool, default=True,
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

parser.add_argument("--micro_batch", "-mc", type=int, default=100,
                    help="Number of micro batches (must evenly divide batch_size)")

parser.add_argument("--dataset_len", "-len", type=int, default=60000,
                    help="Number of dataset, 600000 for MNIST")

parser.add_argument("--latent_dim", "-D", type=int, default=64,
                    help="Embedding dimension")

parser.add_argument("--num_embeddings", "-K", type=int, default=256,
                    help="Number embedding")

parser.add_argument("--dataset", type=str, default="mnist",
                    help="Dataset: mnist, fashion-mnist, cifar10, stl")

parser.add_argument("--recon_num", type=int, default=36, help="Number of reconstruction, must be even")

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
        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings, data_shape)

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
        # grads, _ = self.optimizer._compute_gradients(total_loss, var_list=self.vqvae.trainable_variables, tape=tape)

        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # eps, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        #    60000, 128, 1.3, self.current_epoch * 60000 / 128, 1e-5)
        # print('For delta=1e-5, the current epsilon is: %.2f' % eps)
        # self.epsilon_tracker = tf.keras.metrics.get(eps)

        # self.current_epoch += 1

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
            # "epsilon": eps
        }


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("\nDifferential Privacy Information")

        eps, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
            args.dataset_len, args.batch_size, args.noise_multiplier, epoch+1, args.delta)
        logs["epsilon"] = eps

        checkpoint_path = f"TF_vqvae/{args.dataset}/{'dp' if args.dpsgd else 'normal'}/model"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path += f"/vqvae_cp_{epoch}"

        self.model.save_weights(checkpoint_path)


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

    if args.dpsgd:
        args.dataset_len = train_data.shape[0]
        magnitude = len(str(args.dataset_len))
        args.delta = pow(10, -magnitude)

        optimizer = VectorizedDPAdam(
            l2_norm_clip=args.l2_norm_clip,
            noise_multiplier=args.noise_multiplier,
            num_microbatches=args.micro_batch,
            learning_rate=args.learning_rate
        )
    else:
        optimizer = tf.optimizers.Adam(learning_rate=args.lr)

    vqvae_trainer.compile(optimizer=optimizer)

    history = vqvae_trainer.fit(
        train_data, epochs=args.epochs, batch_size=args.batch_size, callbacks=[CustomCallback()]
    )

    # Save metrics
    vqvae_metric_save_path = f"TF_vqvae/{args.dataset}/{'dp' if args.dpsgd else 'normal'}/metric"
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
    reconstruction_path_traditional_flow = f"TF_vqvae/{args.dataset}/{'dp' if args.dpsgd else 'normal'}/Images/v1"
    if not os.path.exists(reconstruction_path_traditional_flow):
        os.makedirs(reconstruction_path_traditional_flow)

    truth_path = reconstruction_path_traditional_flow + "/grand_truth_images.png"
    trained_vqvae_model = vqvae_trainer.vqvae
    idx = np.random.choice(len(test_data), args.recon_num)
    test_images = test_data[idx]

    # Save grand truth label, if available
    label_save_path = reconstruction_path_traditional_flow + "/grand_truth_label.txt"
    label_names = get_labels(args.dataset)
    val_labels_name = [label_names[i] for i in np.array(test_labels[idx])]
    np.savetxt(label_save_path, val_labels_name, fmt="%s")
    batch_size = int(pow(args.recon_num, 0.5))
    reconstruction_image = trained_vqvae_model.predict(tf.convert_to_tensor(test_images))
    reconstruction_save_path = reconstruction_path_traditional_flow + "/reconstruction_image.png"

    show_batch(test_images, batch_size, truth_path)
    show_batch(reconstruction_image, batch_size, reconstruction_save_path)


    """
    ## Visualizing the discrete codes
    """

    encoder = vqvae_trainer.vqvae.get_layer("encoder")
    quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

    encoded_outputs = encoder.predict(test_images)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

    latent_save_path = reconstruction_path_traditional_flow + "/latent_image.png"
    show_latent(test_images[:6], codebook_indices[:6], reconstruction_image[:6], latent_save_path)

    """
    ## PixelCNN hyperparameters
    """

    '''
    num_residual_blocks = 2
    num_pixelcnn_layers = 2
    pixelcnn_input_shape = encoded_outputs.shape[1:-1]
    print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")

    pixelcnn_inputs = tf.keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, vqvae_trainer.num_embeddings)
    x = PixelConvLayer(
        mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
    )(ohe)

    for _ in range(num_residual_blocks):
        x = ResidualBlock(filters=128)(x)

    for _ in range(num_pixelcnn_layers):
        x = PixelConvLayer(
            mask_type="B",
            filters=128,
            kernel_size=1,
            strides=1,
            activation="relu",
            padding="valid",
        )(x)

    out = layers.Conv2D(
        filters=vqvae_trainer.num_embeddings, kernel_size=1, strides=1, padding="valid"
    )(x)

    pixel_cnn = keras.Model(pixelcnn_inputs, out, name="pixel_cnn")
    # pixel_cnn.summary()

    """
    ## Prepare data to train the PixelCNN

    Objective will be to minimize the CrossEntropy loss between these
    indices and the PixelCNN outputs. Here, the number of categories is equal to the number
    of embeddings present in our codebook (128 in our case). The PixelCNN model is
    trained to learn a distribution (as opposed to minimizing the L1/L2 loss), which is where
    it gets its generative capabilities from.
    """

    # Generate the codebook indices.
    encoded_outputs = encoder.predict(x_train_scaled)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)

    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
    print(f"Shape of the training data for PixelCNN: {codebook_indices.shape}")

    """
    ## PixelCNN training
    """

    pixel_cnn.compile(
        optimizer=keras.optimizers.legacy.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    pixel_cnn.fit(
        x=codebook_indices,
        y=codebook_indices,
        batch_size=128,
        epochs=1,
        validation_split=0.1,
    )

    """
    ## Codebook sampling

    Now that our PixelCNN is trained, we can sample distinct codes from its outputs and pass
    them to our decoder to generate novel images.
    """

    # Create a mini sampler model.
    inputs = layers.Input(shape=pixel_cnn.input_shape[1:])
    outputs = pixel_cnn(inputs, training=False)
    categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
    outputs = categorical_layer(outputs)
    sampler = keras.Model(inputs, outputs)

    """
    We now construct a prior to generate images. Here, we will generate 10 images.
    """

    # Create an empty array of priors.
    batch = 1
    priors = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
    batch, rows, cols = priors.shape

    # Iterate over the priors because generation has to be done sequentially pixel by pixel.
    for row in range(rows):
        for col in range(cols):
            # Feed the whole array and retrieving the pixel value probabilities for the next
            # pixel.
            probs = sampler.predict(priors)
            # Use the probabilities to pick pixel values and append the values to the priors.
            priors[:, row, col] = probs[:, row, col]

    print(f"Prior shape: {priors.shape}")

    """
    We can now use our decoder to generate the images.
    """

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

    for i in range(batch):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i])
        plt.title("Sampling Noise")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated_samples[i].squeeze() + 0.5)
        plt.title("Generated Image")
        plt.axis("off")
        plt.show()
        
    '''


if __name__ == "__main__":
    main()