import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np


# `VectorQuantizer` layer
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
                shape = (self.embedding_dim, self.num_embeddings), dtype = "float32"
                # shape=(self.num_embeddings, self.embedding_dim), dtype="float32"
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

# Encoder and decoder


def get_encoder(latent_dim=128, input_shape=[]):
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(
        encoder_inputs
    )
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    output_channel = input_shape[2]
    # encoder_outputs = layers.Conv2D(latent_dim, output_channel, padding="same")(x)
    encoder_outputs = layers.Conv2D(latent_dim, kernel_size=1, padding="same")(x)

    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def get_decoder(latent_dim=128, input_shape=[]):
    output_channel = input_shape[2]
    latent_inputs = keras.Input(shape=get_encoder(latent_dim, input_shape=input_shape).output.shape[1:])

    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(latent_inputs)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)

    decoder_outputs = layers.Conv2DTranspose(output_channel, 3, strides=1, activation="sigmoid", padding="same")(x)

    return keras.Model(latent_inputs, decoder_outputs, name="decoder")


"""
## Standalone VQ-VAE model
"""


def get_vqvae(latent_dim=16, num_embeddings=64, data_shape=[]):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim, data_shape)
    decoder = get_decoder(latent_dim, data_shape)
    inputs = keras.Input(shape=data_shape)
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")


def get_pixel_cnn(pixelcnn_input_shape, K):
    num_residual_blocks = 3
    num_pixelcnn_layers = 3
    # pixelcnn_input_shape = encoded_outputs.shape[1:-1]
    pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, K)
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

    out = keras.layers.Conv2D(
        filters=K, kernel_size=1, strides=1, padding="valid"
    )(x)

    return keras.Model(pixelcnn_inputs, out, name="pixel_cnn")


# PixelCNN model

# The first layer is the PixelCNN layer. This layer simply
# builds on the 2D convolutional layer, but includes masking.


class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super().__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


# This is just a normal residual block, but based on the PixelConvLayer.
class ResidualBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        # return tf.python.keras.layers.add([inputs, x])
        return layers.add([inputs, x])


# Define the VQ-VAE model
class VQVAE(keras.Model):
    def __init__(self, num_embeddings, embedding_dim, num_latent_vars=128, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_latent_vars = num_latent_vars
        self.beta = beta
        self.encoder = keras.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, activation='relu', padding='same'),
            keras.layers.Conv2D(filters=num_latent_vars, kernel_size=1, strides=1, activation=None, padding='valid')
        ])
        self.decoder = keras.Sequential([
            keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, activation='relu', padding='same'),
            keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, activation='relu', padding='same'),
            keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, activation='relu', padding='same'),
            keras.layers.Conv2DTranspose(filters=3, kernel_size=1, strides=1, activation=None, padding='valid')
        ])
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, beta, name="vector_quantizer")
        # self.vq_layer = VQLayer(num_embeddings, embedding_dim, beta)

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, latent_vars):
        return self.decoder(latent_vars)

    def call(self, inputs):
        # Encode the inputs
        latent_vars = self.encode(inputs)

        # Quantize the latent
        quantized = self.vq_layer(latent_vars)

        # Decode the quantized latent variables
        reconstructed = self.decode(quantized)

        return reconstructed


class VQLayer(keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = self.add_weight(
            shape=(num_embeddings, embedding_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='embedding'
        )

    def call(self, inputs):
        # Reshape inputs to 2D tensor
        flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        # Compute distances between inputs and embeddings
        distances = tf.reduce_sum(tf.square(tf.expand_dims(flat_inputs, axis=1) - self.embedding), axis=2)

        # Find the closest embeddings
        embedding_indices = tf.argmin(distances, axis=1)

        # Convert embedding indices back to original shape
        embedding_indices = tf.reshape(embedding_indices, tf.shape(inputs)[:-1])

        # Look up the closest embeddings
        closest_embeddings = tf.nn.embedding_lookup(self.embedding, embedding_indices)

        # Quantize the input
        # quantized = closest_embeddings + tf.stop_gradient(quantized - closest_embeddings)
        quantized = closest_embeddings + tf.stop_gradient(inputs - closest_embeddings)

        return quantized


