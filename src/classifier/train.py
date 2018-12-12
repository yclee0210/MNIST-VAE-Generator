import argparse
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from vae.vae import Vae
from vae.mmd_vae import MmdVae

import tensorflow.examples.tutorials.mnist as mnist

from config.paths import *


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


if __name__ == '__main__':
    batch_size = 128
    num_classes = 10
    epochs = 12

    FLAGS = parse_args()
    dataset = mnist.input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    x_train_raw, y_train = dataset.train.next_batch(dataset.train.num_examples)
    x_test_raw, y_test = dataset.test.next_batch(dataset.test.num_examples)

    vae_checkpoint_file = build_checkpoint_path(FLAGS.vae_model, FLAGS.latent_dim,
                                                FLAGS.hidden_size,
                                                FLAGS.layers, FLAGS.ckpt)
    classifier_ckpt = build_classifier_checkpoint_path(FLAGS.vae_model, FLAGS.latent_dim,
                                                       FLAGS.hidden_size,
                                                       FLAGS.layers)
    model_class = Vae
    if FLAGS.vae_model == 'mmd':
        model_class = MmdVae

    with tf.Session() as sess:
        vae_model = model_class(FLAGS.latent_dim, FLAGS.hidden_size, FLAGS.layers)
        saver = tf.train.Saver()
        saver.restore(sess, vae_checkpoint_file)

        x_train = vae_model.run(sess, x_train_raw)
        x_test = vae_model.run(sess, x_test_raw)

    # input image dimensions
    img_rows, img_cols = 28, 28

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save(classifier_ckpt)
