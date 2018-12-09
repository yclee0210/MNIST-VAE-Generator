# Decoder class for MNIST dataset
# modified from https://jmetzen.github.io/2015-11-27/vae.html
#
# Created by: Yunchan Clemence Lee

import tensorflow as tf


class Decoder(object):
    def __init__(self, z, n_input, n_hidden_layer_1, n_hidden_layer_2, n_z,
                 transfer_fn):
        self.z = z
        self.weights = {
            'h1': tf.get_variable('decoder_weights_h1', shape=(n_z, n_hidden_layer_1),
                                  dtype=tf.float32),
            'h2': tf.get_variable('decoder_weights_h2', shape=(n_hidden_layer_1, n_hidden_layer_2),
                                  dtype=tf.float32),
            'out_mean': tf.get_variable('decoder_weights_out_mean',
                                        shape=(n_hidden_layer_2, n_input), dtype=tf.float32),
            'out_log_sigma': tf.get_variable('decoder_weights_out_log_sigma',
                                             shape=(n_hidden_layer_2, n_input), dtype=tf.float32)}
        self.biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_layer_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_layer_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}

        layer_1 = transfer_fn(tf.add(tf.matmul(self.z, self.weights['h1']), self.biases['b1']))
        layer_2 = transfer_fn(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))

        self.output = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_2, self.weights['out_mean']), self.biases['out_mean']))
