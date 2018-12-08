# From https://jmetzen.github.io/2015-11-27/vae.html

import argparse
import tensorflow as tf
import numpy as np

from mnist_vae.model import Vae
from mnist_vae.config import config

from scripts.config import *

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae-checkpoint', dest='vae_checkpoint_file',
                        default=MODEL_OUTPUT_FILENAME, type=str)
    parser.add_argument('--save-plot', dest='save_plot', action='store_true')
    return parser.parse_args()


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


def plot(canvas, x, y, save_plot=False):
    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x, y)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    if save_plot:
        results_dir = RESULT_DIR / 'mnist_vae_2d'
        if not results_dir.exists():
            results_dir.mkdir()
        filepath = '%s/%s' % (str(results_dir), '/reconstruct_2d.png')
        plt.savefig(filepath)
    else:
        plt.show()


if __name__ == '__main__':
    args = parse_args()
    architecture = config['mnist_vae_2d']

    with tf.Session() as session:
        vae = Vae(architecture)
        checkpoint_path = build_checkpoint_path('mnist_vae_2d', args.vae_checkpoint_file)
        vae.restore(session, checkpoint_path)

        canvas, x, y = generate_plot(session, vae)
        plot(canvas, x, y, save_plot=args.save_plot)