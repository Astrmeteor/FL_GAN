import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow_gan as tfgan
import tensorflow as tf
import tensorflow_hub as tfhub
import os

current_path = os.path.dirname(os.path.abspath(__file__))


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
        (train_data, train_labels), (test_data, test_labels), info= tfds.as_numpy(
            tfds.load("stl10", split=['train', 'test'], batch_size=-1, as_supervised=True)
        )

    elif dataset == "celeb_a":
        (train_data, test_data), info = tfds.as_numpy(
            tfds.load("celeb_a", split=['train', 'test'], batch_size=-1, as_supervised=True)
        )
        train_labels = None
        test_labels = None

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    assert train_data.min() == 0.
    assert train_data.max() == 1.
    assert test_data.min() == 0.
    assert test_data.max() == 1.
    # assert train_labels.ndim == 1
    # assert test_labels.ndim == 1

    return train_data, train_labels, test_data, test_labels


def show_batch(image_batch, batch_size, save_path, gray=False):
    plt.figure(figsize=(5, 5))
    for n in range(batch_size*batch_size):
        plt.subplot(batch_size, batch_size, n+1)
        if gray:
            plt.imshow(image_batch[n], cmap="gray")
        else:
            plt.imshow(image_batch[n])
        plt.axis("off")
    # plt.show()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def show_latent(ori, latent, recon, save_path, gray=False):
    plt.figure(figsize=(3, 2))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0, right=0.25, top=0.94, bottom=0.1, left=0.12)
    for i in range(len(ori)):
        plt.subplot(10, 3, 1+i*3)
        if gray:
            plt.imshow(ori[i], cmap="gray")
        else:
            plt.imshow(ori[i])
        plt.axis("off")

        plt.subplot(10, 3, 2+i*3)
        if gray:
            plt.imshow(latent[i], cmap="gray")
        else:
            plt.imshow(latent[i])
        plt.axis("off")

        plt.subplot(10, 3, 3+i*3)
        if gray:
            plt.imshow(recon[i], cmap="gray")
        else:
            plt.imshow(recon[i])
        plt.axis("off")
    # plt.show()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def show_sampling(latent, recon, save_path, gray=False):
    plt.figure(figsize=(3, 2))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0, right=0.21, top=1, bottom=0.1, left=0.12)
    for i in range(len(recon)):
        plt.subplot(10, 2, 1 + i * 2)
        if gray:
            plt.imshow(latent[i], cmap="gray")
        else:
            plt.imshow(latent[i])
        plt.axis("off")

        plt.subplot(10, 2, 2 + i * 2)
        if gray:
            plt.imshow(recon[i], cmap="gray")
        else:
            plt.imshow(recon[i])
        plt.axis("off")
    # plt.show()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def get_labels(dataset_name: str = "stl"):
    if dataset_name == "stl":
        label_names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    elif dataset_name == "cifar10":
        label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif dataset_name == "fashion_mnist":
        label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                       "Ankle boot"]
    elif dataset_name == "mnist":
        label_names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    return label_names


@tf.function
def get_inception_score(images):

    image = images[:1000]
    if images.shape[3] == 1:
        image = tf.tile(images[:1000], [1, 1, 1, 3])

    size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
    resized_images = tf.image.resize(image, [size, size], method=tf.image.ResizeMethod.BILINEAR)

    inception_model = tfhub.KerasLayer('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5')
    inc_score = tfgan.eval.inception_score(resized_images[:100], classifier_fn=inception_model)

    return inc_score


# @tf.function
def get_fid_score(real_image, gen_image):

    real_images = real_image
    gen_images = gen_image
    if real_image.shape[3] == 1:
        real_images = tf.tile(real_image, [1, 1, 1, 3])
        gen_images = tf.tile(gen_image, [1, 1, 1, 3])

    size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
    resized_real_images = tf.image.resize(real_images, [size, size], method=tf.image.ResizeMethod.BILINEAR)
    resized_generated_images = tf.image.resize(gen_images, [size, size], method=tf.image.ResizeMethod.BILINEAR)

    resized_real_images = resized_real_images / 127.5 - 1
    resized_generated_images = resized_generated_images / 127.5 - 1

    inception_model = tfhub.KerasLayer('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5')
    fid = tfgan.eval.frechet_inception_distance(resized_real_images, resized_generated_images, classifier_fn=inception_model)

    return fid.numpy()


def get_psnr(real, generated):
    psnr_value = tf.reduce_mean(tf.image.psnr(generated, real, max_val=255.0))
    return psnr_value


# if __name__ == "__main__":
#    load_data("cifar10")