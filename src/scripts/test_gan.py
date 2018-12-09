import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from mnist_vae.model import Vae
from mnist_vae.config import config as vae_config
from latent_gan.config import config as gan_config
from latent_gan.model import LatentGan

from scripts.config import *

import matplotlib.pyplot as plt


def generate_plot(sess, model):
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28 * ny, 28 * nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]] * model.batch_size)
            x_mean = model.decode(sess, z_mu)
            canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

    return canvas, x_values, y_values


if __name__ == "__main__":
    vae_checkpoint = str(CHECKPOINT_DIR / DEFAULT_VAE_CONFIG / MODEL_OUTPUT_FILENAME)
    gan_checkpoint = str(CHECKPOINT_DIR / DEFAULT_GAN_CONFIG / 'model.ckpt-110')
    vae_architecture = vae_config[DEFAULT_VAE_CONFIG]
    gan_architecture = gan_config[DEFAULT_GAN_CONFIG]

    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Session() as session:
        vae = Vae(vae_architecture)
        gan = LatentGan(gan_architecture)
        saver = tf.train.Saver()
        saver.restore(session, gan_checkpoint)

        canvas = np.empty((28 * 20, 28 * 20))
        for i in range(20):
            for j in range(20):
                noise = np.random.normal(size=(gan.batch_size, gan.architecture['n_input']))
                output = session.run(gan.z, feed_dict={gan.x: noise})
                img = vae.decode(session, output)
                canvas[(20 - i - 1) * 28:(20 - i) * 28, j * 28: (j+1) * 28] = img[0].reshape(28, 28)

        plt.figure(figsize=(8, 10))
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.tight_layout()
        results_dir = RESULT_DIR / 'latent_gan'
        if not results_dir.exists():
            results_dir.mkdir()
        filepath = '%s/%s' % (str(results_dir), '/generate.png')
        plt.savefig(filepath)
