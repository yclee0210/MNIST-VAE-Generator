import tensorflow as tf
import numpy as np
from vae.base import BaseVae


class MmdVae(BaseVae):
    def __init__(self, latent_dim=20, hidden_size=500, layers=2, batch_size=100,
                 learning_rate=0.001):
        BaseVae.__init__(self, latent_dim, hidden_size, layers, batch_size, learning_rate)

        self.decoder_loss = -tf.reduce_sum(
            self.x * tf.log(1e-10 + self.x_reconstr_mean) + (1 - self.x) * tf.log(
                1e-10 + 1 - self.x_reconstr_mean), 1)
        gauss = tf.random_normal(tf.stack([batch_size, latent_dim]))
        self.mmd = 1e2 * self.compute_mmd(self.z, gauss)
        self.cost = tf.reduce_mean(self.decoder_loss + self.mmd)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.init = tf.global_variables_initializer()

    def train_batch(self, sess, batch):
        _, cost, mmd, dec_loss = sess.run((self.optimizer, self.cost, self.mmd, self.decoder_loss),
                                          feed_dict={self.x: batch})
        return cost, np.mean(dec_loss), np.mean(mmd)

    def compute_kernel(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(
            -tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

    def run(self, sess, inputs):
        return sess.run(self.x_reconstr_mean, feed_dict={self.x: inputs})
