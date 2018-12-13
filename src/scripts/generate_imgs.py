import argparse
import imageio
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
    return parser.parse_args()


def plot(images, sample=3, result_path='.'):
    canvas = np.ndarray((images.shape[0] * 28, sample * 28))
    for y in range(images.shape[0]):
        sample_img_idxs = np.random.randint(low=0, high=images.shape[1], size=sample)
        for x, i in enumerate(sample_img_idxs):
            canvas[
                (images.shape[0] - y - 1) * 28:(images.shape[0] - y) * 28,
                x * 28:(x + 1) * 28
            ] = images[y, i, :, :]
    imageio.imwrite(str(result_path) + '/samples_medium_2.png', canvas)


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


def generate(vae_model, architecture, gen_model_name, theta=None, n=1, flag=1):
    vae_checkpoint_file = build_checkpoint_path(vae_model, architecture[0],
                                                architecture[1], architecture[2])
    result_path = build_results_path(vae_model, architecture[0],
                                                architecture[1], architecture[2])
    model_class = Vae
    if vae_model == 'mmd':
        model_class = MmdVae

    with tf.Session() as sess:
        vae_model = model_class(architecture[0], architecture[1], architecture[2])
        saver = tf.train.Saver()
        saver.restore(sess, vae_checkpoint_file)

        encodings, labels = get_encodings(sess, vae_model, dataset)
        if gen_model_name == 'simple':
            gen_model = SimpleGenerator(100, decoder=vae_model.decoder)
            gen_model.set_distributions(encodings, labels)
            images = gen_model.generate(sess, stdev=theta, sampling=n)
        elif flag == 1:
            gen_model = MediumGenerator(100, decoder=vae_model.decoder)
            gen_model.set_distributions(encodings, labels)
            images = gen_model.generate(sess, stdevs=theta, sampling=n)
        else:
            gen_model = MediumGenerator(100, decoder=vae_model.decoder)
            gen_model.set_distributions(encodings, labels)
            images = gen_model.generate2(sess, stdevs=theta, sampling=n)
        plot(images, sample=10, result_path=result_path)


if __name__ == '__main__':
    FLAGS = parse_args()
    dataset = mnist.input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # generate('vae', (20, 500, 2), 'simple', theta=0.4, n=2)
    # generate('mmd', (20, 500, 2), 'simple', theta=0.6, n=3)
    # generate('mmd', (10, 500, 2), 'medium', theta=np.array([0., 0., 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8]), n=1)
    generate('mmd', (10, 500, 2), 'medium', theta=np.array([
        [0., 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.4, 0.4, 1.],
        [0., 0., 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.8],
        [0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.5, 0.5, 0.5, 1.],
        [0., 0., 0., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9],
        [0., 0., 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.8],
        [0., 0.1, 0.3, 0.3, 0.3, 0.4, 0.5, 0.5, 0.5, 1.],
        [0., 0., 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.4, 1.],
        [0., 0.2, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 1.],
        [0., 0.1, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 1.],
    ]), n=1, flag=2)












# 0 [0.  0.1 0.1 0.1 0.1 0.2 0.2 0.4 0.4 1. ] 0.7889660000801086
# 1 [0.  0.  0.2 0.3 0.3 0.3 0.3 0.4 0.4 0.8] 0.6693890929222107
# 2 [0.1 0.1 0.1 0.1 0.4 0.4 0.5 0.5 0.5 1. ] 0.6845813304185867
# 3 [0.  0.  0.  0.5 0.5 0.5 0.5 0.5 0.5 0.9] 0.6783482819795609
# 4 [0.  0.  0.4 0.4 0.4 0.5 0.5 0.5 0.5 0.8] 0.7537929952144623
# 5 [0.  0.1 0.3 0.3 0.3 0.4 0.5 0.5 0.5 1. ] 0.5949949592351913
# 6 [0.  0.  0.1 0.5 0.5 0.5 0.5 0.5 0.5 1. ] 0.6897886514663696
# 7 [0.1 0.1 0.1 0.1 0.1 0.1 0.4 0.4 0.4 1. ] 0.7326874256134033
# 8 [0.  0.2 0.4 0.4 0.5 0.5 0.5 0.5 0.5 1. ] 0.7380033850669860
# 9 [0.  0.1 0.3 0.3 0.3 0.4 0.4 0.4 0.4 1. ] 0.7447232365608215
# 0.7075275358557701 0.7075275358557701 1.0
# 1544662638.769147

# [0., 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.4, 0.4, 1.],
# [0., 0., 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.8],
# [0., 0., 0., 0.1, 0.1, 0.1, 0.3, 0.4, 0.4, 0.9],
# [0., 0., 0., 0.1, 0.1, 0.3, 0.3, 0.4, 0.4, 1.],
# [0., 0., 0., 0.2, 0.2, 0.2, 0.3, 0.3, 0.4, 1.],
# [0., 0., 0., 0., 0.1, 0.1, 0.3, 0.3, 0.4, 1.],
# [0., 0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.4, 1.],
# [0., 0., 0., 0., 0.2, 0.2, 0.3, 0.3, 0.4, 1.],
# [0., 0., 0., 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 1.],
# [0., 0., 0.1, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 1.]
# 0.7460929474234581 0.7460929474234581 1.0
# 1544654449.227783
