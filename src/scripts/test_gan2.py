import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from mnist_vae.model import Vae
from mnist_vae.config import config as vae_config
from latent_gan.config import config as gan_config
from latent_gan.model import LatentGan
from latent_gan_2.model import LatentGan2

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


def run_vae(session, vae, dataset, vae_architecture):
    num_batch = int(dataset.train.num_examples / 100)
    labels = None
    z_means = None
    for i in range(num_batch):
        data, batch_labels = dataset.train.next_batch(100)
        batch_z_means = session.run(vae.z, feed_dict={vae.x: data})
        if labels is None:
            labels = batch_labels
        else:
            labels = np.concatenate([labels, batch_labels])
        if z_means is None:
            z_means = batch_z_means
        else:
            z_means = np.concatenate([z_means, batch_z_means])

    means = list()
    stds = list()
    for i in range(10):
        indices = np.where(labels[:, i] == 1)[0]
        z_mean_i = np.take(z_means, indices, axis=0)
        means_i = list()
        stds_i = list()
        for j in range(vae_architecture['n_z']):
            means_i.append(np.mean(z_mean_i[:, j]))
            stds_i.append(np.std(z_mean_i[:, j]))
        means.append(means_i)
        stds.append(stds_i)

    means = np.array(means, dtype=np.float32)
    stds = np.array(stds, dtype=np.float32)

    return means, stds


if __name__ == "__main__":
    vae_checkpoint = str(CHECKPOINT_DIR / DEFAULT_VAE_CONFIG / MODEL_OUTPUT_FILENAME)
    vae_architecture = vae_config[DEFAULT_VAE_CONFIG]

    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Session() as session:
        vae = Vae(vae_architecture)
        vae.restore(session, vae_checkpoint)

        means, stds = run_vae(session, vae, dataset, vae_architecture)

        gan = LatentGan2(means, stds, 100, vae.decoder)
        fake_numbers = np.random.randint(0, 10, size=100)
        fake_labels = np.zeros(shape=(100, 10))
        fake_labels[np.arange(100), fake_numbers] = 1
        mean_x = np.matmul(fake_labels, means)
        std_x = np.matmul(fake_labels, stds)
        eps = np.random.normal(0, 0.5, size=(100, 20, 3))
        z = np.mean([mean_x + std_x * eps[:, :, i] for i in range(3)], axis=0)
        decoded = session.run(gan.decoder.output, feed_dict={gan.decoder.z: z})
        generated = gan.generator.predict(decoded)
        canvas = np.empty((28 * 6, 28 * 4))
        for i in range(3):
            for j in range(4):
                canvas[(6 - (2 * i) - 1) * 28:(6 - (2 * i)) * 28, j * 28:(j + 1) * 28] = generated[
                                                                                         4 * i + j,
                                                                                         :, :, 0]
                canvas[(6 - (2 * i + 1) - 1) * 28:(6 - (2 * i + 1)) * 28, j * 28:(j + 1) * 28] = \
                decoded[4 * i + j].reshape((28, 28))
        plt.figure(figsize=(4, 4))
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.tight_layout()
        plt.savefig('start_sample.png')
        n_samples = dataset.train.num_examples
        for epoch in range(10):
            avg_dis_cost = [0., 0.]
            avg_adv_cost = [0., 0.]
            num_batches = int(n_samples / vae.batch_size)
            for i in range(num_batches):
                x, y = dataset.train.next_batch(100)
                dis_cost, adv_cost = gan.train(session, means, stds, x, y)
                avg_dis_cost[0] += dis_cost[0] / n_samples * vae.batch_size
                avg_adv_cost[0] += adv_cost[0] / n_samples * vae.batch_size
                avg_dis_cost[1] += dis_cost[1] / n_samples * vae.batch_size
                avg_adv_cost[1] += adv_cost[1] / n_samples * vae.batch_size
                print(i, '|', num_batches, end='\r')
            print(epoch, avg_dis_cost, avg_adv_cost)
            fake_numbers = np.random.randint(0, 10, size=100)
            fake_labels = np.zeros(shape=(100, 10))
            fake_labels[np.arange(100), fake_numbers] = 1
            mean_x = np.matmul(fake_labels, means)
            std_x = np.matmul(fake_labels, stds)
            eps = np.random.normal(0, 0.5, size=(100, 20, 3))
            z = np.mean([mean_x + std_x * eps[:, :, i] for i in range(3)], axis=0)
            decoded = session.run(gan.decoder.output, feed_dict={gan.decoder.z: z})
            generated = gan.generator.predict(decoded)
            canvas = np.empty((28 * 6, 28 * 4))
            for i in range(3):
                for j in range(4):
                    canvas[(6 - (2 * i) - 1) * 28:(6 - (2 * i)) * 28, j * 28:(j + 1) * 28] = generated[4*i + j, :, :, 0]
                    canvas[(6 - (2 * i + 1) - 1) * 28:(6 - (2 * i + 1)) * 28, j * 28:(j + 1) * 28] = decoded[4*i + j].reshape((28, 28))
            plt.figure(figsize=(4, 4))
            plt.imshow(canvas, origin="upper", cmap="gray")
            plt.tight_layout()
            plt.savefig('epoch_%d_sample.png' % epoch)

        gan.generator.save('generator.h5')
        gan.discriminator.save('discriminator.h5')

        # results_dir = RESULT_DIR / 'latent_gan'
        # if not results_dir.exists():
        #     results_dir.mkdir()
        # filepath = '%s/%s' % (str(results_dir), '/generate.png')
        # plt.savefig(filepath)
