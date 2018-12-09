# GAN that generates latent vector z and discriminates

import tensorflow as tf
import numpy as np


class LatentGan(object):
    def __init__(self, architecture, transfer_fn=tf.nn.softplus, learning_rate=0.001,
                 batch_size=100):
        self.architecture = architecture
        self.transfer_fn = transfer_fn
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Generator takes a random input noise
        self.x = tf.placeholder(tf.float32, [None, self.architecture['n_input']])
        self.y = tf.placeholder(tf.float32, [None, 2])

        self._create_network()
        self._create_optimizer()

        self.init = tf.global_variables_initializer()

    def _create_network(self):
        self.generator = Generator(self.x, self.architecture['n_input'],
                                   self.architecture['n_hidden_gen_layer_1'],
                                   self.architecture['n_hidden_gen_layer_2'],
                                   self.architecture['n_z'], self.transfer_fn)
        self.z = self.generator.output
        self.discriminator = Discriminator(self.z, self.architecture['n_z'],
                                           self.architecture['n_hidden_dis_layer_1'],
                                           self.architecture['n_hidden_dis_layer_2'],
                                           tf.nn.softplus)
        self.prediction = self.discriminator.output

    def _create_optimizer(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,
                                                               logits=self.prediction)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss)

    def _get_saver(self):
        return tf.train.Saver()

    def train_batch(self, sess, batch):
        # Generate fake inputs
        noise = np.random.normal(size=(self.batch_size, self.architecture['n_input']))
        fake = sess.run(self.z, feed_dict={self.x: noise})

        half = int(self.batch_size / 2)
        z = np.concatenate((batch[:half, :], fake[:half, :]))
        y = np.ones([self.batch_size, 2])
        y[half:, 0] = 0
        y[:half, 1] = 0

        _, dis_cost1 = sess.run((self.discriminator.optimizer, self.discriminator.loss),
                                feed_dict={self.x: noise, self.discriminator.z_direct: z,
                                           self.discriminator.y: y})
        z = np.concatenate((batch[half:, :], fake[half:, :]))
        _, dis_cost2 = sess.run((self.discriminator.optimizer, self.discriminator.loss),
                                feed_dict={self.x: noise, self.discriminator.z_direct: z,
                                           self.discriminator.y: y})

        y = np.ones([self.batch_size, 2])
        y[:, 0] = 0
        x = np.random.normal(size=(self.batch_size, self.architecture['n_input']))
        _, adv_cost = sess.run((self.optimizer, self.loss),
                               feed_dict={self.x: x, self.y: y})

        return np.concatenate((dis_cost1, dis_cost2)), adv_cost

    def train(self, sess, real, epochs=10, display_step=5, checkpoint_file=None):
        sess.run(self.init)

        n_real = real.shape[0]
        num_batches = int(n_real / self.batch_size)

        # Save option
        saver = None
        if checkpoint_file is not None:
            saver = self._get_saver()

        for epoch in range(epochs):
            batch_xs = real[np.random.randint(0, real.sahpe[0], size=self.batch_size), :, :]
            for i in range(num_batches):
                dis_cost, adv_cost = self.train_batch(sess, batch_xs)

                if (epoch + 1) % display_step == 0:
                    print("Epoch %04d: discriminator cost=%.9f, GAN cost=%.9f" % (
                        epoch + 1, dis_cost, adv_cost))
        if checkpoint_file is not None:
            saver.save(sess, checkpoint_file)

    def generate(self):
        pass

    def evaluate(self):
        pass

    def restore(self, sess, checkpoint):
        saver = self._get_saver()
        saver.restore(sess, checkpoint)


class Generator(object):
    def __init__(self, x, n_input, n_hidden_layer_1, n_hidden_layer_2, n_z, transfer_fn):
        self.x = x
        self.weights = {
            'h1': tf.get_variable('gen_weights_h1', shape=(n_input, n_hidden_layer_1),
                                  dtype=tf.float32),
            'h2': tf.get_variable('gen_weights_h2', shape=(n_hidden_layer_1, n_hidden_layer_2),
                                  dtype=tf.float32),
            'out': tf.get_variable('gen_weights_out',
                                   shape=(n_hidden_layer_2, n_z), dtype=tf.float32)}
        self.biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_layer_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_layer_2], dtype=tf.float32)),
            'out': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}

        # 2 Softplus layers
        layer_1 = transfer_fn(tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1']))
        layer_2 = transfer_fn(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))

        # Regularize?
        self.output = tf.add(tf.matmul(tf.nn.sigmoid(layer_2), self.weights['out']), self.biases['out'])


class Discriminator(object):
    def __init__(self, z, n_z, n_hidden_layer_1, n_hidden_layer_2, transfer_fn):
        self.z = z
        self.y = tf.placeholder(tf.float32, [None, 2])
        self.z_direct = tf.placeholder(tf.float32, [None, n_z])
        self.weights = {
            'h1': tf.get_variable('dis_weights_h1', shape=(n_z, n_hidden_layer_1),
                                  dtype=tf.float32),
            'h2': tf.get_variable('dis_weights_h2', shape=(n_hidden_layer_1, n_hidden_layer_2),
                                  dtype=tf.float32),
            'out': tf.get_variable('dis_weights_out',
                                   shape=(n_hidden_layer_2, 2), dtype=tf.float32),
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_layer_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_layer_2], dtype=tf.float32)),
            'out': tf.Variable(tf.zeros([2], dtype=tf.float32))}

        layer_1 = transfer_fn(tf.add(tf.matmul(self.z, self.weights['h1']), self.biases['b1']))
        layer_2 = transfer_fn(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))

        dis_layer_1 = transfer_fn(
            tf.add(tf.matmul(self.z_direct, self.weights['h1']), self.biases['b1']))
        dis_layer_2 = transfer_fn(
            tf.add(tf.matmul(dis_layer_1, self.weights['h2']), self.biases['b2']))

        # logit output
        self.output = tf.add(tf.matmul(layer_2, self.weights['out']), self.biases['out'])
        self.output_direct = tf.add(tf.matmul(dis_layer_2, self.weights['out']), self.biases['out'])

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.output)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
