# Decoder class for MNIST dataset
# modified from https://jmetzen.github.io/2015-11-27/vae.html
#
# Created by: Yunchan Clemence Lee

import tensorflow as tf


class Decoder(object):
    def __init__(self, z, hidden_size, layers, img_dim):
        self.input = z
        self.weights = {
            'h1': tf.get_variable('decoder_weights_h1', shape=(z.shape[1], hidden_size),
                                  dtype=tf.float32),
            'h2': tf.get_variable('decoder_weights_h2', shape=(hidden_size, hidden_size),
                                  dtype=tf.float32),
            'out_mean': tf.get_variable('decoder_weights_out_mean',
                                        shape=(hidden_size, img_dim), dtype=tf.float32),
            'out_log_sigma': tf.get_variable('decoder_weights_out_log_sigma',
                                             shape=(hidden_size, img_dim), dtype=tf.float32)}
        self.biases = {
            'b1': tf.Variable(tf.zeros([hidden_size], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([hidden_size], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([img_dim], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([img_dim], dtype=tf.float32))}

        layer_1 = tf.nn.softplus(
            tf.add(tf.matmul(self.input, self.weights['h1']), self.biases['b1']))
        layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))

        self.output = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_2, self.weights['out_mean']), self.biases['out_mean']))
