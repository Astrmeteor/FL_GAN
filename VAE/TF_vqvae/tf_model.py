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

# Encoder and decoder


def get_encoder(latent_dim=16, input_shape=[]):
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(latent_dim, 3, activation="relu", strides=2, padding="same")(
        encoder_inputs
    )
    x = layers.Conv2D(latent_dim, 3, activation="relu", strides=2, padding="same")(x)
    output_channel = input_shape[2]
    encoder_outputs = layers.Conv2D(latent_dim, output_channel, padding="same")(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def get_decoder(latent_dim=16, input_shape=[]):
    latent_inputs = keras.Input(shape=get_encoder(latent_dim, input_shape=input_shape).output.shape[1:])
    x = layers.Conv2DTranspose(latent_dim, 3, activation="relu", strides=2, padding="same")(
        latent_inputs
    )
    x = layers.Conv2DTranspose(latent_dim, 3, activation="relu", strides=2, padding="same")(x)
    output_channel = input_shape[2]
    decoder_outputs = layers.Conv2DTranspose(output_channel, 3, padding="same")(x)
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


"""
# To-do
class GatedPixelCnn(tf.keras.Model):
    def __init__(self, K, in_channels=64, n_layers=15, n_filters=256):
        super(GatedPixelCnn, self).__init__(name="gated_pixel_cnn")
        self.embedding = tf.keras.layers.Embedding(K, in_channels)
        self.in_conv = PixelConvLayer(mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same")
    
    def train_step(self, data):
        # input conv layer
        # logger.info("Building CONV_IN")
        net = conv(self.inputs, conf.gated_conv_num_feature_maps, [7, 7], "A", num_channels, scope="CONV_IN")

        # main gated layers
        for idx in xrange(conf.gated_conv_num_layers):
            scope = 'GATED_CONV%d' % idx
            net = gated_conv(net, [3, 3], num_channels, scope=scope)
            logger.info("Building %s" % scope)

        # output conv layers
        net = tf.nn.relu(conv(net, conf.output_conv_num_feature_maps, [1, 1], "B", num_channels, scope='CONV_OUT0'))
        logger.info("Building CONV_OUT0")
        self.logits = tf.nn.relu(
            conv(net, q_levels * num_channels, [1, 1], "B", num_channels, scope='CONV_OUT1'))  # shape [N,H,W,DC]
        logger.info("Building CONV_OUT1")

        if (num_channels > 1):
            self.logits = tf.reshape(self.logits, [-1, height, width, q_levels,
                                                   num_channels])  # shape [N,H,W,DC] -> [N,H,W,D,C]
            self.logits = tf.transpose(self.logits,
                                       perm=[0, 1, 2, 4, 3])  # shape [N,H,W,D,C] -> [N,H,W,C,D]

        flattened_logits = tf.reshape(self.logits, [-1, q_levels])  # [N,H,W,C,D] -> [NHWC,D]
        target_pixels_loss = tf.reshape(self.target_pixels, [-1])  # [N,H,W,C] -> [NHWC]

"""




