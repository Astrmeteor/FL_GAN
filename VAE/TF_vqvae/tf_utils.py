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


def show_batch(image_batch, batch_size, save_path):
    plt.figure(figsize=(5, 5))
    for n in range(batch_size*batch_size):
        ax = plt.subplot(batch_size, batch_size, n+1)
        plt.imshow(image_batch[n])
        plt.axis("off")
    # plt.show()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def show_latent(ori, latent, recon, save_path):
    plt.figure(figsize=(3, 2))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0, right=0.25, top=0.94, bottom=0.1, left=0.12)
    for i in range(len(ori)):
        plt.subplot(10, 3, 1+i*3)
        plt.imshow(ori[i])
        plt.axis("off")

        plt.subplot(10, 3, 2+i*3)
        plt.imshow(latent[i])
        plt.axis("off")

        plt.subplot(10, 3, 3+i*3)
        plt.imshow(recon[i])
        plt.axis("off")
    # plt.show()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def show_sampling(latent, recon, save_path):
    plt.figure(figsize=(3, 2))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0, right=0.21, top=1, bottom=0.1, left=0.12)
    for i in range(len(recon)):
        plt.subplot(10, 2, 1 + i * 2)
        plt.imshow(latent[i])
        plt.axis("off")

        plt.subplot(10, 2, 2 + i * 2)
        plt.imshow(recon[i])
        plt.axis("off")
    # plt.show()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def get_labels(dataset_name: str = "stl"):
    if dataset_name == "stl":
        label_names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    elif dataset_name == "cifar":
        label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif dataset_name == "fashion-mnist":
        label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                       "Ankle boot"]
    elif dataset_name == "mnist":
        label_names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    return label_names

# if __name__ == "__main__":
#    load_data("cifar10")