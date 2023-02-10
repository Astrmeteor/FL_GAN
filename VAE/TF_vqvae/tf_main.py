import absl.app
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf

import tensorflow_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_vectorized import VectorizedDPAdam
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from keras import layers

from tf_model import get_vqvae, show_subplot
from tf_utils import load_data

from absl import flags
from absl import logging


"""
## Wrapping up the training loop inside `VQVAETrainer`
"""


class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, latent_dim=64, num_embeddings=128, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = get_vqvae(self.latent_dim, self.num_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

        # self.current_epoch = 0
        # self.epsilon_tracker = keras.metrics.get(name="eps")

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
            FLAGS.dataset_len, FLAGS.batch_size, FLAGS.noise_multiplier, epoch, FLAGS.delta)

# Flags

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
                    'train with vanilla SGD.')

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training', short_name='lr')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_float('noise_multiplier', 1.3, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_integer('epochs', 5, 'Number of epochs')
flags.DEFINE_float('delta', 1e-5, 'Delta')
flags.DEFINE_integer('batch_size', 256, 'Number of batch size')
flags.DEFINE_integer('micro_batches', 100, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_integer('number_dataset', 0, 'Number of dataset', short_name='num')
flags.DEFINE_integer('latent_dim', 64, 'Embedding dimension', short_name='D')
flags.DEFINE_integer('num_embeddings', 256, 'Number embedding', short_name='K')
flags.DEFINE_string('dataset', 'mnist', 'dataset: mnist, fashion-mnist, cifar10, stl')
flags.DEFINE_integer('dataset_len', 60000, 'Number of dataset')

FLAGS = flags.FLAGS


def main(argv):
    """
    ## Load dataset
    """
    train_data, train_labels, test_data, test_labels = load_data(FLAGS.dataset)
    data_variance = np.var(train_data, dtype=np.float32)

    """
    ## Train the VQ-VAE model
    """

    vqvae_trainer = VQVAETrainer(data_variance, latent_dim=FLAGS.D, num_embeddings=FLAGS.K)

    if FLAGS.dpsgd:
        optimizer = VectorizedDPAdam(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            num_microbatches=FLAGS.micro_batches,
            learning_rate=FLAGS.lr
        )
    else:
        optimizer = tf.optimizers.Adam(learning_rate=FLAGS.lr)

    vqvae_trainer.compile(optimizer=optimizer)
    history = vqvae_trainer.fit(
        train_data, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, callbacks=[CustomCallback()]
    )

    """
    ## Reconstruction results on the test set
    """

    trained_vqvae_model = vqvae_trainer.vqvae
    idx = np.random.choice(len(test_data), 1)
    test_images = train_data[idx]
    reconstructions_test = trained_vqvae_model.predict(tf.convert_to_tensor(test_images))

    for test_image, reconstructed_image in zip(test_images, reconstructions_test):
        show_subplot(test_image, reconstructed_image)

    """
    ## Visualizing the discrete codes
    """

    encoder = vqvae_trainer.vqvae.get_layer("encoder")
    quantizer = vqvae_trainer.vqvae.get_layer("vector_quantizer")

    encoded_outputs = encoder.predict(test_images)
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

    for i in range(len(test_images)):
        plt.subplot(1, 2, 1)
        plt.imshow(test_images[i].squeeze() + 0.5, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(codebook_indices[i], cmap="gray")
        plt.title("Code")
        plt.axis("off")
        plt.show()

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
    absl.app.run(main)