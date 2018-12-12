import argparse
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist as mnist

from config.paths import *
from vae.vae import Vae
from vae.mmd_vae import MmdVae


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', default=str(DATA_ROOT), type=str)
    parser.add_argument('--vae_model', dest='vae_model', default='vae', type=str)
    parser.add_argument('--latent_dim', dest='latent_dim', default=20, type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=100, type=int)
    parser.add_argument('--display_step', dest='display_step', default=10, type=int)
    parser.add_argument('--hidden_size', dest='hidden_size', default=500, type=int)
    parser.add_argument('--layers', dest='layers', default=2, type=int)
    parser.add_argument('--epochs', dest='epochs', default=75, type=int)
    return parser.parse_args()


def train():
    checkpoint_file = build_checkpoint_path(FLAGS.vae_model, FLAGS.latent_dim, FLAGS.hidden_size,
                                            FLAGS.layers)
    model_class = Vae
    if FLAGS.vae_model == 'mmd':
        model_class = MmdVae

    with tf.Session() as sess:
        model = model_class(FLAGS.latent_dim, FLAGS.hidden_size, FLAGS.layers)
        sess.run(model.init)

        # define dataset size & batchs per epoch to run
        n_samples = dataset.train.num_examples
        num_batches = int(n_samples / FLAGS.batch_size)

        # Save option
        saver = None
        if checkpoint_file is not None:
            saver = tf.train.Saver()

        # for each epoch
        for epoch in range(FLAGS.epochs):
            # initialize avg cost
            avg_cost = 0.
            avg_decoder_loss = 0.
            avg_other_loss = 0.

            # run each batch
            for i in range(num_batches):
                batch_xs, _ = dataset.train.next_batch(FLAGS.batch_size)
                cost, decoder_loss, other_loss = model.train_batch(sess, batch_xs)
                avg_cost += cost / n_samples * FLAGS.batch_size
                avg_decoder_loss += decoder_loss / n_samples * FLAGS.batch_size
                avg_other_loss += other_loss / n_samples * FLAGS.batch_size

            # Output log for every 5 epochs & also save a checkpoint if defined
            if (epoch + 1) % FLAGS.display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost =", "{:.9f}".format(avg_cost),
                      "decl =", "{:.9f}".format(avg_decoder_loss),
                      "othr =", "{:.9f}".format(avg_other_loss))
                if checkpoint_file is not None:
                    saver.save(sess, '%s-%d' % (checkpoint_file, epoch + 1))
        saver.save(sess, checkpoint_file)


if __name__ == '__main__':
    FLAGS = parse_args()
    dataset = mnist.input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    train()
