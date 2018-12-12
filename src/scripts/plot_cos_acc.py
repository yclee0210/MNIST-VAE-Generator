import argparse
import numpy as np

from config.paths import *

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_model', dest='vae_model', default='vae', type=str)
    parser.add_argument('--gen_model', dest='gen_model', default='simple', type=str)
    parser.add_argument('--latent_dim', dest='latent_dim', default=20, type=int)
    parser.add_argument('--hidden_size', dest='hidden_size', default=500, type=int)
    parser.add_argument('--layers', dest='layers', default=2, type=int)
    parser.add_argument('--ckpt_num', dest='ckpt_num', type=int)
    return parser.parse_args()


if __name__ == "__main__":
    FLAGS = parse_args()

    results_path = build_results_path('vae', FLAGS.latent_dim, FLAGS.hidden_size,
                                      FLAGS.layers, FLAGS.ckpt_num)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in [1,2,3,5]:
        data = np.loadtxt('%s/simple_%d' % (results_path, i))[:, 1:]
        ax.scatter(data[:, 1], data[:, 2], marker='x', label=('%s_%d' % ('vae', i)))

    results_path = build_results_path('mmd', FLAGS.latent_dim, FLAGS.hidden_size,
                                      FLAGS.layers, FLAGS.ckpt_num)

    for i in [1, 2, 3, 5]:
        data = np.loadtxt('%s/simple_%d' % (results_path, i))[:, 1:]
        ax.scatter(data[:, 1], data[:, 2], marker='o', label=('%s_%d' % ('mmd', i)))
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
