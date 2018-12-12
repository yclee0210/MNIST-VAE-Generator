import tensorflow as tf
import numpy as np
from vae.encoder import Encoder
from vae.decoder import Decoder

dim = 28


class BaseVae(object):
    def __init__(self, latent_dim=20, hidden_size=500, layers=2, batch_size=100,
                 learning_rate=0.001):
        self.batch_size = batch_size

        with tf.name_scope('data'):
            self.x = tf.placeholder(tf.float32, [None, dim * dim])

        with tf.variable_scope('encoder'):
            self.encoder = Encoder(x=self.x, latent_dim=latent_dim, hidden_size=hidden_size,
                                   layers=layers)
            self.z_mean, self.z_log_sigma_sq = self.encoder.output
            eps = tf.random_normal(tf.shape(self.z_mean), 0, 1, dtype=tf.float32)
            self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        with tf.variable_scope('decoder'):
            self.decoder = Decoder(z=self.z, hidden_size=hidden_size, layers=layers,
                                   img_dim=(dim * dim))
            self.x_reconstr_mean = self.decoder.output

    def encode(self, sess, dataset, mean_only=False):
        batch, labels = dataset.test.next_batch(5000)
        if mean_only:
            return sess.run(self.z_mean,  feed_dict={self.x: batch}), labels
        else:
            return sess.run(self.z,  feed_dict={self.x: batch}), labels

    def decode(self, sess, z_mean):
        return sess.run(self.x_reconstr_mean, feed_dict={self.z: z_mean})

    def generate(self, sess):
        z_mean = np.random.normal(size=self.architecture['n_z'])
        return self.decode(sess, z_mean)

    def run(self, sess, inputs):
        return sess.run(self.x_reconstr_mean, feed_dict={self.x: inputs})
