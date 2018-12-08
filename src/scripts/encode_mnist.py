# Encode mnist to latent vectors

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


def encode(sess, model, xs):
    return model.encode(sess, xs)


def plot_data(coord, color, config=DEFAULT_VAE_CONFIG, save_plot=False):
    plt.figure(figsize=(8, 6))
    plt.scatter(coord[:, 0], coord[:, 1], c=np.argmax(color, 1))
    plt.colorbar()
    plt.grid()

    if save_plot:
        results_dir = RESULT_DIR / config
        if not results_dir.exists():
            results_dir.mkdir()
        filepath = '%s/%s' % (str(results_dir), '/encode_mnist.png')
        plt.savefig(filepath)
    else:
        plt.show()


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

        z_mean, labels = encode(session, vae, dataset)

        if args.plot:
            plot_data(z_mean, labels, config=args.config, save_plot=args.save_plot)
