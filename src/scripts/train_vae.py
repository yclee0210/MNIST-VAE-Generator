# Training script for mnist vae

import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from mnist_vae.model import Vae
from mnist_vae.config import config

from scripts.config import *

DEFAULT_CKPT = str(CHECKPOINT_DIR / DEFAULT_VAE_CONFIG / MODEL_OUTPUT_FILENAME)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default=DEFAULT_VAE_CONFIG, type=str)
    parser.add_argument('--output', dest='model_output', default=MODEL_OUTPUT_FILENAME, type=str)
    parser.add_argument('--epochs', dest='epochs', default=DEFAULT_EPOCHS, type=int)
    return parser.parse_args()


def train(sess, model, epochs=DEFAULT_EPOCHS, checkpoint_file=None):
    model.train(sess, dataset, epochs=epochs, checkpoint_file=checkpoint_file)


if __name__ == '__main__':
    checkpoint_path = DEFAULT_CKPT
    args = parse_args()

    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Session() as session:
        if args.config in config:
            checkpoint_path = build_checkpoint_path(args.config, args.model_output)
            architecture = config[args.config]
        else:
            raise ValueError('Configuration %s does not exist' % args.config)

        vae = Vae(architecture)
        train(session, vae, epochs=args.epochs, checkpoint_file=checkpoint_path)
