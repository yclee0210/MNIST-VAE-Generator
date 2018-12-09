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
        self.label = tf.placeholder(tf.float32, [None, self.architecture['n_label']])
        self.one_hots = np.eye(self.architecture['n_label'])

        self._create_network()
        self._create_optimizer()

        self.init = tf.global_variables_initializer()

    def _create_network(self):
        self.generator = Generator(self.x, self.label, self.architecture['n_input'],
                                   self.architecture['n_label'],
                                   self.architecture['n_hidden_gen_input_layer_1'],
                                   self.architecture['n_hidden_gen_label_layer_1'],
                                   self.architecture['n_hidden_gen_layer_2'],
                                   self.architecture['n_z'], self.transfer_fn)
        self.z = self.generator.output
        self.discriminator = Discriminator(self.z, self.y, self.label, self.architecture['n_z'],
                                           self.architecture['n_label'],
                                           self.architecture['n_hidden_dis_input_layer_1'],
                                           self.architecture['n_hidden_dis_label_layer_1'],
                                           self.architecture['n_hidden_dis_layer_2'],
                                           self.transfer_fn)
        self.prediction = self.discriminator.output

    def _create_optimizer(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,
                                                               logits=self.prediction)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss, var_list=self.generator.variables)

    def _get_saver(self):
        return tf.train.Saver()

    def train_batch(self, sess, batch, label):
        # Generate fake inputs
        noise = np.random.normal(size=(self.batch_size, self.architecture['n_input']))
        fake_label = np.zeros((noise.shape[0], self.architecture['n_label']))
        for i in range(len(label)):
            fake_label[i, :] = self.one_hots[
                np.random.randint(low=0, high=self.architecture['n_label'])]
        fake = sess.run(self.z, feed_dict={self.x: noise, self.label: fake_label})

        half = int(self.batch_size / 2)
        y = np.ones([self.batch_size, 2])
        y[half:, 0] = 0
        y[:half, 1] = 0

        z = np.concatenate((batch[:half, :], fake[:half, :]))
        labels = np.concatenate((label[:half, :], np.zeros([half, label.shape[1]])))
        _, dis_cost1 = sess.run((self.discriminator.optimizer, self.discriminator.loss),
                                feed_dict={self.discriminator.z_direct: z,
                                           self.label: labels, self.y: y})

        z = np.concatenate((batch[half:, :], fake[half:, :]))
        labels = np.concatenate((label[half:, :], np.zeros([half, label.shape[1]])))
        _, dis_cost2 = sess.run((self.discriminator.optimizer, self.discriminator.loss),
                                feed_dict={self.discriminator.z_direct: z,
                                           self.label: labels, self.y: y})

        x = np.random.normal(size=(self.batch_size, self.architecture['n_input']))
        labels = np.zeros((noise.shape[0], self.architecture['n_label']))
        for i in range(len(label)):
            labels[i, :] = self.one_hots[
                np.random.randint(low=0, high=self.architecture['n_label'])]
        y = np.ones([self.batch_size, 2])
        y[:, 1] = 0
        _, adv_cost = sess.run((self.optimizer, self.loss),
                               feed_dict={self.x: x, self.label: labels, self.y: y})

        return np.concatenate((dis_cost1, dis_cost2)), adv_cost

    def generate(self):
        pass

    def evaluate(self):
        pass

    def restore(self, sess, checkpoint):
        saver = self._get_saver()
        saver.restore(sess, checkpoint)


class Generator(object):
    def __init__(self, x, label, n_input, n_label, n_hidden_input_layer_1, n_hidden_label_layer_1,
                 n_hidden_layer_2, n_z, transfer_fn):
        self.x = x
        self.label = label
        self.weights = {
            'input1': tf.get_variable('gen_weights_i1', shape=(n_input, n_hidden_input_layer_1),
                                      dtype=tf.float32),
            'label1': tf.get_variable('gen_weights_l1', shape=(n_label, n_hidden_label_layer_1),
                                      dtype=tf.float32),
            'h2': tf.get_variable('gen_weights_h2',
                                  shape=(n_hidden_input_layer_1 + n_hidden_label_layer_1,
                                         n_hidden_layer_2),
                                  dtype=tf.float32),
            'out': tf.get_variable('gen_weights_out', shape=(n_hidden_layer_2, n_z),
                                   dtype=tf.float32)}
        self.biases = {
            'bi1': tf.Variable(tf.zeros([n_hidden_input_layer_1], dtype=tf.float32)),
            'bl1': tf.Variable(tf.zeros([n_hidden_label_layer_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_layer_2], dtype=tf.float32)),
            'out': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}

        # 2 Softplus layers
        input_layer_1 = transfer_fn(
            tf.add(tf.matmul(self.x, self.weights['input1']), self.biases['bi1']))
        label_layer_1 = transfer_fn(
            tf.add(tf.matmul(self.label, self.weights['label1']), self.biases['bl1']))
        concat = tf.concat([input_layer_1, label_layer_1], 1)
        layer_2 = transfer_fn(tf.add(tf.matmul(concat, self.weights['h2']), self.biases['b2']))

        self.output = tf.add(tf.matmul(tf.nn.sigmoid(layer_2), self.weights['out']),
                             self.biases['out'])
        self.variables = [var for key, var in self.weights.items()] + [var for key, var in
                                                                       self.biases.items()]
        # z_mean = tf.add(tf.matmul(layer_2, self.weights['out_mean']), self.biases['out_mean'])
        # z_log_sigma_sq = tf.add(tf.matmul(layer_2, self.weights['out_log_sigma']),
        #                         self.biases['out_log_sigma'])
        # Regularize?
        # eps = tf.random_normal((100, n_z), 0, 1, dtype=tf.float32)
        # self.output = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))


class Discriminator(object):
    def __init__(self, z, y, label, n_z, n_label, n_hidden_input_layer_1, n_hidden_label_layer_1,
                 n_hidden_layer_2, transfer_fn):
        self.z = z
        self.label = label
        self.y = y
        self.z_direct = tf.placeholder(tf.float32, [None, n_z])
        self.weights = {
            'input1': tf.get_variable('dis_weights_i1', shape=(n_z, n_hidden_input_layer_1),
                                      dtype=tf.float32),
            'label1': tf.get_variable('dis_weights_l1', shape=(n_label, n_hidden_label_layer_1),
                                      dtype=tf.float32),
            'h2': tf.get_variable('dis_weights_h2',
                                  shape=(n_hidden_input_layer_1 + n_hidden_label_layer_1,
                                         n_hidden_layer_2),
                                  dtype=tf.float32),
            'out': tf.get_variable('dis_weights_out',
                                   shape=(n_hidden_layer_2, 2), dtype=tf.float32),
        }
        self.biases = {
            'bi1': tf.Variable(tf.zeros([n_hidden_input_layer_1], dtype=tf.float32)),
            'bl1': tf.Variable(tf.zeros([n_hidden_label_layer_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_layer_2], dtype=tf.float32)),
            'out': tf.Variable(tf.zeros([2], dtype=tf.float32))}

        input_layer_1 = transfer_fn(
            tf.add(tf.matmul(self.z, self.weights['input1']), self.biases['bi1']))
        label_layer_1 = transfer_fn(
            tf.add(tf.matmul(self.label, self.weights['label1']), self.biases['bl1']))
        concat = tf.concat([input_layer_1, label_layer_1], 1)
        layer_2 = transfer_fn(tf.add(tf.matmul(concat, self.weights['h2']), self.biases['b2']))

        dis_input_layer_1 = transfer_fn(
            tf.add(tf.matmul(self.z_direct, self.weights['input1']), self.biases['bi1']))
        concat2 = tf.concat([dis_input_layer_1, label_layer_1], 1)
        dis_layer_2 = transfer_fn(
            tf.add(tf.matmul(concat2, self.weights['h2']), self.biases['b2']))

        # logit output
        self.output = tf.add(tf.matmul(layer_2, self.weights['out']), self.biases['out'])
        self.output_direct = tf.add(tf.matmul(dis_layer_2, self.weights['out']), self.biases['out'])

        self.variables = [var for key, var in self.weights.items()] + [var for key, var in
                                                                       self.biases.items()]
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,
                                                               logits=self.output_direct)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss,
                                                                              var_list=self.variables)
