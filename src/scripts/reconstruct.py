import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from vae.vae import Vae
from vae.mmd_vae import MmdVae

from config.paths import *

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', default=str(DATA_ROOT), type=str)
    parser.add_argument('--vae_model', dest='vae_model', default='vae', type=str)
    parser.add_argument('--latent_dim', dest='latent_dim', default=20, type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=100, type=int)
    parser.add_argument('--display_step', dest='display_step', default=10, type=int)
    parser.add_argument('--hidden_size', dest='hidden_size', default=500, type=int)
    parser.add_argument('--layers', dest='layers', default=2, type=int)
    parser.add_argument('--ckpt_num', dest='ckpt_num', type=int)
    return parser.parse_args()


def reconstruct(sess, model, samples):
    x_sample = samples.test.next_batch(100)[0]

    return x_sample, model.run(sess, x_sample)


def plot(sample, reconstruction, config_name, result_path=None):
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
    if result_path is not None:
        results_dir = RESULT_ROOT / config_name
        if not results_dir.exists():
            results_dir.mkdir()
        filepath = '%s/%s' % (str(result_path), '/reconstruction.png')
        plt.savefig(filepath)
    else:
        plt.show()


if __name__ == '__main__':
    FLAGS = parse_args()

    checkpoint_path = build_checkpoint_path(FLAGS.vae_model, FLAGS.latent_dim, FLAGS.hidden_size,
                                            FLAGS.layers, FLAGS.ckpt_num)
    results_path = build_results_path(FLAGS.vae_model, FLAGS.latent_dim, FLAGS.hidden_size,
                                            FLAGS.layers, FLAGS.ckpt_num)

    dataset = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    model_class = Vae
    if FLAGS.vae_model == 'mmd':
        model_class = MmdVae
    with tf.Session() as session:
        vae = model_class(FLAGS.latent_dim, FLAGS.hidden_size, FLAGS.layers)
        saver = tf.train.Saver()
        saver.restore(session, checkpoint_path)

        inputs, outputs = reconstruct(session, vae, dataset)
        plot(inputs, outputs, FLAGS.vae_model, results_path)
