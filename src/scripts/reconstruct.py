import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from mnist_vae.model import Vae
from mnist_vae.config import config

from scripts.config import *

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default=DEFAULT_VAE_CONFIG, type=str)
    parser.add_argument('--vae-checkpoint', dest='vae_checkpoint_file',
                        default=MODEL_OUTPUT_FILENAME, type=str)
    parser.add_argument('--save-plot', dest='save_plot', action='store_true')
    return parser.parse_args()


def reconstruct(sess, model, samples):
    x_sample = samples.test.next_batch(100)[0]

    return x_sample, model.run(sess, x_sample)


def plot(sample, reconstruction, config_name, save=False):
    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(reconstruction[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    if save:
        results_dir = RESULT_DIR / config_name
        if not results_dir.exists():
            results_dir.mkdir()
        filepath = '%s/%s' % (str(results_dir), '/reconstruction.png')
        plt.savefig(filepath)
    else:
        plt.show()


if __name__ == '__main__':
    args = parse_args()
    if not args.config or not args.vae_checkpoint_file:
        raise ValueError('config or checkpoint file missing')

    architecture = config[args.config]
    checkpoint_path = build_checkpoint_path(args.config, args.vae_checkpoint_file)

    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Session() as session:
        vae = Vae(architecture)
        vae.restore(session, checkpoint_path)

        inputs, outputs = reconstruct(session, vae, dataset)
        plot(inputs, outputs, args.config, args.save_plot)