import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from mnist_vae.model import Vae
from mnist_vae.config import config as vae_config
from latent_gan.config import config as gan_config
from latent_gan.model import LatentGan

from scripts.config import *


if __name__ == "__main__":
    vae_checkpoint = str(CHECKPOINT_DIR / DEFAULT_VAE_CONFIG / MODEL_OUTPUT_FILENAME)
    gan_checkpoint = str(CHECKPOINT_DIR / DEFAULT_GAN_CONFIG / MODEL_OUTPUT_FILENAME)
    vae_architecture = vae_config[DEFAULT_VAE_CONFIG]
    gan_architecture = gan_config[DEFAULT_GAN_CONFIG]

    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Session() as session:
        vae = Vae(vae_architecture)
        vae.restore(session, vae_checkpoint)

        gan = LatentGan(gan_architecture)
        session.run(gan.init)
        saver = tf.train.Saver()

        n_samples = dataset.train.num_examples
        for epochs in range(500):
            avg_dis_cost = 0.
            avg_adv_cost = 0.
            for i in range(int(n_samples / vae.batch_size)):
                data, labels = dataset.train.next_batch(vae.batch_size)
                encoded = session.run(vae.z, feed_dict={vae.x: data})
                dis_cost, adv_cost = gan.train_batch(session, encoded, labels)
                avg_dis_cost += np.mean(dis_cost) / n_samples * vae.batch_size
                avg_adv_cost += np.mean(adv_cost) / n_samples * vae.batch_size
            if epochs % 10 == 0:
                print(epochs, avg_dis_cost, avg_adv_cost)
                saver.save(session, gan_checkpoint + '-%d' % epochs)
                print('Checkpoint saved')
