import tensorflow as tf

from tf_model import get_pixel_cnn

num_embeddings = 256
pixel_cnn = get_pixel_cnn(pixelcnn_input_shape, num_embeddings)