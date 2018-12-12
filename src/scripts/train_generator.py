import argparse
import keras
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist as mnist

from config.paths import *
from vae.vae import Vae
from vae.mmd_vae import MmdVae
from generator.simple import SimpleGenerator
from generator.medium import MediumGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', default=str(DATA_ROOT), type=str)
    parser.add_argument('--vae_model', dest='vae_model', default='vae', type=str)
    parser.add_argument('--gen_model', dest='gen_model', default='simple', type=str)
    parser.add_argument('--latent_dim', dest='latent_dim', default=20, type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=100, type=int)
    parser.add_argument('--display_step', dest='display_step', default=10, type=int)
    parser.add_argument('--hidden_size', dest='hidden_size', default=500, type=int)
    parser.add_argument('--layers', dest='layers', default=2, type=int)
    parser.add_argument('--epochs', dest='epochs', default=75, type=int)
    parser.add_argument('--ckpt', dest='ckpt', type=str)
    return parser.parse_args()


def get_encodings(sess, model, dataset):
    num_batch = int(dataset.train.num_examples / model.batch_size)
    ys = None
    z_means = None
    for i in range(num_batch):
        data, batch_ys = dataset.train.next_batch(model.batch_size)
        batch_z_means = sess.run(model.z, feed_dict={model.x: data})
        if ys is None:
            ys = batch_ys
        else:
            ys = np.concatenate([ys, batch_ys])
        if z_means is None:
            z_means = batch_z_means
        else:
            z_means = np.concatenate([z_means, batch_z_means])
    return z_means, ys


def train(dataset):
    vae_checkpoint_file = build_checkpoint_path(FLAGS.vae_model, FLAGS.latent_dim,
                                                FLAGS.hidden_size,
                                                FLAGS.layers, FLAGS.ckpt)
    classifier_ckpt = build_classifier_checkpoint_path(FLAGS.vae_model, FLAGS.latent_dim,
                                                       FLAGS.hidden_size,
                                                       FLAGS.layers)
    result_path = build_results_path(FLAGS.vae_model, FLAGS.latent_dim, FLAGS.hidden_size,
                                     FLAGS.layers)
    model_class = Vae
    if FLAGS.vae_model == 'mmd':
        model_class = MmdVae

    with tf.Session() as sess:
        vae_model = model_class(FLAGS.latent_dim, FLAGS.hidden_size, FLAGS.layers)
        saver = tf.train.Saver()
        saver.restore(sess, vae_checkpoint_file)

        encodings, labels = get_encodings(sess, vae_model, dataset)
        if FLAGS.gen_model == 'simple':
            gen_model = SimpleGenerator(FLAGS.batch_size, decoder=vae_model.decoder)
        else:
            gen_model = MediumGenerator(FLAGS.batch_size, decoder=vae_model.decoder)
        classifier = keras.models.load_model(classifier_ckpt)
        sampling = [1, 2, 3, 5]
        if FLAGS.gen_model == 'simple':
            tables = gen_model.train(sess, encodings, labels, classifier, sampling=sampling,
                                     batches=10)

            for i in range(tables.shape[0]):
                print(tables[i, :, :])
                np.savetxt('%s/simple_%d' % (result_path, sampling[i]), tables[i, :, :])
        elif FLAGS.gen_model == 'medium1':
            cost, vals = gen_model.train(sess, encodings, labels, classifier)
            print(cost)
            print(vals)
            np.savetxt('%s/medium' % result_path, vals)
        else:
            cost, vals = gen_model.train2(sess, encodings, labels, classifier)
            print(cost)
            print(vals)
            np.savetxt('%s/medium' % result_path, vals)
            # [0.  0.  0.  0.  0.  0.  0.2 0.3 0.3 0.9]
            # 0.8624786180257797 0.9989999999999999 0.8724786180257809
            # [0.  0.  0.  0.  0.  0.2 0.2 0.3 0.3 1. ]
            # 0.8612541016936301 0.9989999999999999 0.8712541016936313
            # [0.  0.  0.  0.  0.1 0.1 0.4 0.4 0.4 1. ]
            # 0.8168011233210564 0.9959999999999999 0.8568011233210575
            # [0.  0.  0.1 0.1 0.3 0.3 0.3 0.3 0.3 1. ]
            # 0.8434233599901200 0.9989999999999999 0.8534233599901211
            # [0.  0.  0.2 0.2 0.4 0.4 0.4 0.4 0.4 0.8]
            # 0.8132053801417352 0.9979999999999999 0.8332053801417363




if __name__ == '__main__':
    FLAGS = parse_args()
    train(mnist.input_data.read_data_sets(FLAGS.data_dir, one_hot=True))
