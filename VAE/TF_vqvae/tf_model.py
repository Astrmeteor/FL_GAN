import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, LeakyReLU, Reshape
import numpy as np

# `VectorQuantizer` layer
class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape = (self.embedding_dim, self.num_embeddings), dtype = "float32"
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
    # encoder_outputs = layers.Conv2D(latent_dim, output_channel, padding="same")(x)
    encoder_outputs = layers.Conv2D(latent_dim, kernel_size=1, padding="same")(x)

    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def get_decoder(latent_dim=128, input_shape=[]):
    output_channel = input_shape[2]
    latent_inputs = keras.Input(shape=get_encoder(latent_dim, input_shape=input_shape).output.shape[1:])

    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(latent_inputs)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)

    decoder_outputs = layers.Conv2DTranspose(output_channel, 3, activation="sigmoid", strides=1, padding="same")(x)

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


class VAE(keras.Model):
    def __init__(self, dataset):
        super(VAE, self).__init__()

        assert dataset in ["mnist", "fashion-mnist", "cifar", "stl"]

        # latent features
        self.n_latent_features = 128

        # resolution
        if dataset in ["mnist", "fashion-mnist"]:
            pooling_kernel = [2, 2]
            encoder_output_size = 7
        elif dataset == "cifar":
            pooling_kernel = [4, 2]
            encoder_output_size = 4
        elif dataset == "stl":
            pooling_kernel = [4, 4]
            encoder_output_size = 6

        # color channels
        if dataset in ["mnist", "fashion-mnist"]:
            color_channels = 1
        else:
            color_channels = 3

        # neurons int middle layer
        n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size

        # Encoder
        self.encoder = Encoder(color_channels, pooling_kernel, n_neurons_middle_layer)
        # Middle
        self.fc1 = Dense(self.n_latent_features)
        self.fc2 = Dense(self.n_latent_features)
        self.fc3 = Dense(n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(color_channels, pooling_kernel, encoder_output_size)

        # history
        self.history = {"loss": [], "val_loss": []}

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=mu.shape)
        return eps * tf.exp(logvar * .5) + mu

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def sampling(self, x=64):
        z = tf.random.normal(shape=(x, self.n_latent_features))
        z = self.fc3(z)
        return self.decoder(z)

    def call(self, inputs):
        h = self.encoder(inputs)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar


class EncoderModule(keras.Model):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = Conv2D(output_channels, kernel_size=kernel, padding='same' if pad else 'valid', strides=stride)
        self.bn = BatchNormalization()
        self.relu = LeakyReLU()

    def call(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(keras.Model):
    def __init__(self, color_channels, pooling_kernels, n_neurons_in_middle_layer):
        super().__init__()
        self.n_neurons_in_middle_layer = n_neurons_in_middle_layer
        self.bottle = EncoderModule(color_channels, 32, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(32, 64, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(64, 128, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = EncoderModule(128, 256, stride=pooling_kernels[1], kernel=3, pad=1)

    def call(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        return tf.reshape(out, (-1, self.n_neurons_in_middle_layer))


class DecoderModule(keras.Model):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.convt = Conv2DTranspose(input_channels, kernel_size=stride, strides=stride)
        self.bn = BatchNormalization()

        if activation == "relu":
            self.activation = LeakyReLU()
        elif activation == "sigmoid":
            self.activation = Activation('sigmoid')
        elif activation == "tanh":
            self.activation = Activation('tanh')

    def call(self, x):
        return self.activation(self.bn(self.convt(x)))


class Decoder(keras.Model):
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        super().__init__()
        self.decoder_input_size = decoder_input_size
        self.m1 = DecoderModule(256, 128, stride=1)
        self.m2 = DecoderModule(128, 64, stride=pooling_kernels[1])
        self.m3 = DecoderModule(64, 32, stride=pooling_kernels[0])
        self.bottle = DecoderModule(32, color_channels, stride=1, activation="tanh")

    def call(self, x):
        out = tf.reshape(x, (-1, 256, self.decoder_input_size, self.decoder_input_size))
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)
