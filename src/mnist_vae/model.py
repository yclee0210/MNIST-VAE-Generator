# Variational Autoencoder for MNIST dataset
# modified from https://jmetzen.github.io/2015-11-27/vae.html
#
# Created by: Yunchan Clemence Lee

import tensorflow as tf
import numpy as np
from mnist_vae.encoder import Encoder
from mnist_vae.decoder import Decoder


class Vae(object):
    def __init__(self, network_architecture, transfer_fn=tf.nn.softplus, learning_rate=0.001,
                 batch_size=100):
        self.architecture = network_architecture
        self.transfer_fn = transfer_fn
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Input variable
        self.x = tf.placeholder(tf.float32, [None, self.architecture['n_input']])

        self._create_network()
        self._create_optimizer()

        self.init = tf.global_variables_initializer()

    def _create_network(self):
        # Initialize encoder
        self.encoder = Encoder(self.x, self.architecture['n_input'],
                               self.architecture['n_hidden_encoder_layer_1'],
                               self.architecture['n_hidden_encoder_layer_2'],
                               self.architecture['n_z'], self.transfer_fn)
        self.z_mean, self.z_log_sigma_sq = self.encoder.output

        # Define intermediate
        n_z = self.architecture['n_z']
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Initialize deecoder
        self.decoder = Decoder(self.z, self.architecture['n_input'],
                               self.architecture['n_hidden_decoder_layer_1'],
                               self.architecture['n_hidden_decoder_layer_2'],
                               self.architecture['n_z'], self.transfer_fn)
        self.x_reconstr_mean = self.decoder.output

    def _create_optimizer(self):
        decoder_loss = -tf.reduce_sum(
            self.x * tf.log(1e-10 + self.x_reconstr_mean) + (1 - self.x) * tf.log(
                1e-10 + 1 - self.x_reconstr_mean), 1)
        kl_divergence = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)

        self.cost = tf.reduce_mean(decoder_loss + kl_divergence)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.cost)

    def _get_saver(self):
        return tf.train.Saver({
                self.encoder.weights['h1'].name: self.encoder.weights['h1'],
                self.encoder.weights['h2'].name: self.encoder.weights['h2'],
                self.encoder.weights['out_mean'].name: self.encoder.weights['out_mean'],
                self.encoder.weights['out_log_sigma'].name: self.encoder.weights['out_log_sigma'],
                self.encoder.biases['b1'].name: self.encoder.biases['b1'],
                self.encoder.biases['b2'].name: self.encoder.biases['b2'],
                self.encoder.biases['out_mean'].name: self.encoder.biases['out_mean'],
                self.encoder.biases['out_log_sigma'].name: self.encoder.biases['out_log_sigma'],
                self.decoder.weights['h1'].name: self.decoder.weights['h1'],
                self.decoder.weights['h2'].name: self.decoder.weights['h2'],
                self.decoder.weights['out_mean'].name: self.decoder.weights['out_mean'],
                self.decoder.weights['out_log_sigma'].name: self.decoder.weights['out_log_sigma'],
                self.decoder.biases['b1'].name: self.decoder.biases['b1'],
                self.decoder.biases['b2'].name: self.decoder.biases['b2'],
                self.decoder.biases['out_mean'].name: self.decoder.biases['out_mean'],
                self.decoder.biases['out_log_sigma'].name: self.decoder.biases['out_log_sigma'],
            })

    def train_batch(self, sess, batch):
        _, cost = sess.run((self.optimizer, self.cost), feed_dict={self.x: batch})
        return cost

    def train(self, sess, dataset, epochs=10, display_step=5, checkpoint_file=None):
        # initialize variables
        sess.run(self.init)

        # define dataset size & batchs per epoch to run
        n_samples = dataset.train.num_examples
        num_batches = int(n_samples / self.batch_size)

        # Save option
        saver = None
        if checkpoint_file is not None:
            saver = self._get_saver()

        # for each epoch
        for epoch in range(epochs):
            # initialize avg cost
            avg_cost = 0.

            # run each batch
            for i in range(num_batches):
                batch_xs, _ = dataset.train.next_batch(self.batch_size)
                cost = self.train_batch(sess, batch_xs)
                avg_cost += cost / n_samples * self.batch_size

            # Output log for every 5 epochs & also save a checkpoint if defined
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost=", "{:.9f}".format(avg_cost))
                if checkpoint_file is not None:
                    saver.save(sess, '%s-%d' % (checkpoint_file, epoch + 1))
        saver.save(sess, checkpoint_file)

    def encode(self, sess, dataset):
        batch, labels = dataset.test.next_batch(5000)
        return sess.run(self.z_mean, feed_dict={self.x: batch}), labels

    def decode(self, sess, z_mean):
        return sess.run(self.x_reconstr_mean, fead_dict={self.z: z_mean})

    def generate(self, sess):
        z_mean = np.random.normal(size=self.architecture['n_z'])
        return self.decode(sess, z_mean)

    def restore(self, sess, checkpoint):
        saver = self._get_saver()
        saver.restore(sess, checkpoint)
