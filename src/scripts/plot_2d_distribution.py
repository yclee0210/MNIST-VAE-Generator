# Encode mnist to latent vectors

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


def plot_data(coord, color, results_path=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(coord[:, 0], coord[:, 1], c=np.argmax(color, 1))
    plt.colorbar()
    plt.grid()

    if results_path is not None:
        filepath = '%s/%s' % (results_path, '/2d_distribution.png')
        plt.savefig(filepath)
    else:
        plt.show()


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

        z_mean, labels = vae.encode(session, dataset)
        plot_data(z_mean, labels, results_path)
