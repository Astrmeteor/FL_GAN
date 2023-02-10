import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.python.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf


# VectorQuantizer layer
class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
                tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
                + tf.reduce_sum(self.embeddings ** 2, axis=0)
                - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


class ResidualLayer(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.resblock = keras.Sequential(
            keras.layers.BatchNormalization(dim),
            keras.layers.Activation(activation="relu"),
            keras.layers.Conv2D(filters=dim, kernel_size=3, strides=1, padding="same")
        )

    def call(self, x):
        return self.resblock(x) + x


class Encoder(layers.Layer):
    def __init__(self, D=256, **kwargs):
        super().__init__(**kwargs)
        self.net = keras.Sequential(
            layers.Conv2D(filters=D, kernel_size=3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(filters=D, kernel_size=3, strides=2, padding="same", activation="relu"),
            ResidualLayer(D),
            ResidualLayer(D)
        )

    def call(self, x):
        return self.net(x)


class Decoder(layers.Layer):
    def __init__(self, D=256, output_channel = 3, **kwargs):
        super().__init__(**kwargs)
        self.net = keras.Sequential(
            ResidualLayer(D),
            ResidualLayer(D),
            keras.layers.BatchNormalization(D),
            keras.layers.Activation(activation="relu"),
            layers.Conv2DTranspose(input_dim=D, filters=D, strides=2, padding="same"),
            keras.layers.BatchNormalization(D),
            keras.layers.Activation(activation="relu"),
            layers.Conv2DTranspose(input_dim=D, filters=output_channel, strides=2, padding="same")
        )

    def call(self, x):
        return self.net(x)


class VQVAE(keras.models.Model):
    def __init__(self, K, D, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.D = D

        self.codebook = VectorQuantizer(self.K, self.D)
        self.encoder = Encoder(D=self.D)
        self.decoder = Decoder(D=self.D)

    def call(self, x):
        z_e = self.encoder(x)
        q, z_q = self.codebook(z_e)
        x_reconstructed = self.decoder(z_e)
        return x_reconstructed, z_e, z_q


if __name__ == "__main__":
    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    vqvae = VQVAE(K=128, D=256)
    vqvae.compile(optimizer=keras.optimizers.Adam())
    vqvae.fit(x_train, epochs=2, batch_size=64)




