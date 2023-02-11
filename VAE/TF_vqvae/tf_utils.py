import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def load_data(dataset):

    if dataset == "mnist":
        # train, test = tf.keras.datasets.mnist.load_data()
        (train_data, train_labels), (test_data, test_labels) = tfds.as_numpy(
            tfds.load("mnist", split=['train', 'test'], batch_size=-1, as_supervised=True)
        )

    elif dataset == "fashion_mnist":
        # train, test = tf.keras.datasets.fashion_mnist.load_data()
        (train_data, train_labels), (test_data, test_labels) = tfds.as_numpy(
            tfds.load("fashion_mnist", split = ['train', 'test'], batch_size=-1, as_supervised=True)
        )

    elif dataset == "cifar10":
        # train, test = tf.keras.datasets.cifar10.load_data()
        (train_data, train_labels), (test_data, test_labels) = tfds.as_numpy(
            tfds.load("cifar10", split=['train', 'test'], batch_size=-1, as_supervised=True)
        )

    elif dataset == "stl":
        (train_data, train_labels), (test_data, test_labels) = tfds.as_numpy(
            tfds.load("stl10", split=['train', 'test'], batch_size=-1, as_supervised=True)
        )

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    assert train_data.min() == 0.
    assert train_data.max() == 1.
    assert test_data.min() == 0.
    assert test_data.max() == 1.
    assert train_labels.ndim == 1
    assert test_labels.ndim == 1

    return train_data, train_labels, test_data, test_labels


def show_subplot(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()

# if __name__ == "__main__":
#    load_data("cifar10")