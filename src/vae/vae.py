import tensorflow as tf
import numpy as np
from vae.base import BaseVae


class Vae(BaseVae):
    def __init__(self, latent_dim=20, hidden_size=500, layers=2, batch_size=100,
                 learning_rate=0.001):
        BaseVae.__init__(self, latent_dim, hidden_size, layers, batch_size, learning_rate)

        self.decoder_loss = -tf.reduce_sum(
            self.x * tf.log(1e-10 + self.x_reconstr_mean) + (1 - self.x) * tf.log(
                1e-10 + 1 - self.x_reconstr_mean), 1)
        self.kl_divergence = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(self.decoder_loss + self.kl_divergence)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.init = tf.global_variables_initializer()

    def train_batch(self, sess, batch):
        _, cost, kl, dec_loss = sess.run((self.optimizer, self.cost, self.kl_divergence, self.decoder_loss),
                           feed_dict={self.x: batch})
        return cost, np.mean(dec_loss), np.mean(kl)

    def run(self, sess, inputs):
        return sess.run(self.x_reconstr_mean, feed_dict={self.x: inputs})
