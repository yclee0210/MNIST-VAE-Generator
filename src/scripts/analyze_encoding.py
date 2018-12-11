# Analyze encoding

import argparse
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from mnist_vae.model import Vae
from mnist_vae.config import config

from scripts.config import *
from scripts.train_vae import train as train_vae

import matplotlib.pyplot as plt

DEFAULT_CKPT = str(CHECKPOINT_DIR / DEFAULT_VAE_CONFIG / MODEL_OUTPUT_FILENAME)
DEFAULT_FIGURE_DIR = RESULT_DIR / DEFAULT_VAE_CONFIG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default=DEFAULT_VAE_CONFIG, type=str)
    parser.add_argument('--vae-checkpoint', dest='vae_checkpoint_file', type=str)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--save-plot', dest='save_plot', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    checkpoint_path = DEFAULT_CKPT
    args = parse_args()

    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Session() as session:
        if args.config in config:
            checkpoint_path = build_checkpoint_path(args.config, args.vae_checkpoint_file)
            architecture = config[args.config]
        else:
            raise ValueError('Configuration %s does not exist' % args.config)

        vae = Vae(architecture)
        if args.vae_checkpoint_file:
            checkpoint_path = build_checkpoint_path(args.config, args.vae_checkpoint_file)
            vae.restore(session, checkpoint_path)
        else:
            checkpoint_path = build_checkpoint_path(args.config, MODEL_OUTPUT_FILENAME)
            train_vae(session, vae)

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

        stats = list()
        for i in range(10):
            indices = np.where(labels[:, i] == 1)[0]
            z_mean_i = np.take(z_means, indices, axis=0)
            stats_i = list()
            for j in range(architecture['n_z']):
                stat = (np.mean(z_mean_i[:, j]), np.std(z_mean_i[:, j]))
                stats_i.append(stat)
            stats.append(stats_i)

        canvas = np.empty((28 * 10, 28 * 10))
        for i in range(10):
            stat = stats[i]
            # z = np.array([[mean for mean, std in stat] for i in range(100)])
            # z = np.array([[mean + std * np.random.normal(0, 1) for mean, std in stat] for i in range(100)])
            # z = np.array([[mean + std * np.random.normal(0, 0.1) for mean, std in stat] for i in range(100)])
            z = np.array([np.mean(np.array([[mean + std * np.random.normal(0, 0.5) for mean, std in stat] for i in range(3)]), 0) for i in range(100)])
            print(z.shape)

            out = session.run(vae.x_reconstr_mean, feed_dict={vae.z: z})
            print(out.shape)
            for j in range(10):
                pic = out[j]
                canvas[(10 - i - 1) * 28:(10 - i) * 28, j * 28:(j + 1) * 28] = out[j].reshape(28, 28)

        plt.figure(figsize=(8, 10))
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.tight_layout()
        plt.show()
                # if args.plot:
        #     plot_data(z_mean, labels, config=args.config, save_plot=args.save_plot)
