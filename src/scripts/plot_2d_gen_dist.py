# From https://jmetzen.github.io/2015-11-27/vae.html

import argparse
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from vae.vae import Vae
from vae.mmd_vae import MmdVae

from config.paths import *

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', default=str(DATA_ROOT), type=str)
    parser.add_argument('--vae_model', dest='vae_model', default='vae', type=str)
    parser.add_argument('--batch_size', dest='batch_size', default=100, type=int)
    parser.add_argument('--display_step', dest='display_step', default=10, type=int)
    parser.add_argument('--hidden_size', dest='hidden_size', default=500, type=int)
    parser.add_argument('--layers', dest='layers', default=2, type=int)
    parser.add_argument('--ckpt_num', dest='ckpt_num', type=int)
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


def plot(canvas, x, y, results_path):
    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x, y)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    filepath = '%s/%s' % (results_path, '/generate_2d.png')
    plt.savefig(filepath)


if __name__ == '__main__':
    FLAGS = parse_args()

    checkpoint_path = build_checkpoint_path(FLAGS.vae_model, 2, FLAGS.hidden_size,
                                            FLAGS.layers, FLAGS.ckpt_num)
    results_path = build_results_path(FLAGS.vae_model, 2, FLAGS.hidden_size,
                                            FLAGS.layers, FLAGS.ckpt_num)
    dataset = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    model_class = Vae
    if FLAGS.vae_model == 'mmd':
        model_class = MmdVae
    with tf.Session() as session:
        vae = model_class(2, FLAGS.hidden_size, FLAGS.layers)
        saver = tf.train.Saver()
        saver.restore(session, checkpoint_path)

        canvas, x, y = generate_plot(session, vae)
        plot(canvas, x, y, results_path)
