import argparse

import tensorflow as tf
import numpy as np
import imageio
from tensorflow.examples.tutorials.mnist import input_data

from scripts.config import *

from mnist_vae.model import Vae
from mnist_vae.config import config

'''
install dependencies (including imageio)
run from src dir
python -m scripts.sample_generation --batches=10
python -m scripts.sample_generation --batches=10 sampling=2
python -m scripts.sample_generation --batches=10 sampling=3
python -m scripts.sample_generation --batches=10 sampling=5
python -m scripts.sample_generation --batches=10 sampling=10
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default=DEFAULT_VAE_CONFIG, type=str)
    parser.add_argument('--ckpt-file', dest='ckpt_file', default=MODEL_OUTPUT_FILENAME, type=str)
    parser.add_argument('--step-size', dest='step_size', default=0.1, type=float)
    parser.add_argument('--sampling', dest='sampling', default=1, type=int)
    parser.add_argument('--batches', dest='batches', default=1, type=int)
    return parser.parse_args()


def get_encodings(sess, model, dataset):
    num_batch = int(dataset.train.num_examples / model.batch_size)
    ys = None
    z_means = None
    for i in range(num_batch):
        data, batch_ys = dataset.train.next_batch(model.batch_size)
        batch_z_means = sess.run(vae.z, feed_dict={vae.x: data})
        if ys is None:
            ys = batch_ys
        else:
            ys = np.concatenate([ys, batch_ys])
        if z_means is None:
            z_means = batch_z_means
        else:
            z_means = np.concatenate([z_means, batch_z_means])
    return z_means, ys


def get_distributions(zs, ys):
    result = np.ndarray((ys.shape[1], zs.shape[1], 2))
    for y in range(ys.shape[1]):
        idxs = np.where(ys[:, y] == 1)[0]
        zs_cond_y = np.take(zs, idxs, axis=0)
        for dim in range(zs.shape[1]):
            feature_values = zs_cond_y[:, dim]
            result[y, dim, 0] = np.mean(feature_values)
            result[y, dim, 1] = np.std(feature_values)
    return result


def gen_param_results(sess, model, distributions, stdev=0, sampling=1, batches=1, save_path=None):
    results = np.empty((distributions.shape[0], model.batch_size * batches, 28, 28))
    for y in range(distributions.shape[0]):
        save_to = save_path / ('label_%d' % y)
        if not save_to.exists():
            save_to.mkdir()
        dist = distributions[y]
        imgs = gen_imgs(sess, model, dist, stdev, sampling, batches)
        results[y, :, :, :] = imgs
        np.save(save_to / 'imgs', imgs)
    return results


def gen_imgs(sess, model, dist, stdev, sampling, batches):
    xs = np.empty((model.batch_size * batches, 28, 28))
    for i in range(batches):
        eps = np.random.normal(scale=stdev,
                               size=(model.batch_size, distributions.shape[1], sampling))
        eps_mean = np.mean(eps, axis=2)
        z_mean = dist[:, 0]
        z_stdv = dist[:, 1]
        zs = z_mean + z_stdv * eps_mean
        gen_x = sess.run(model.decoder.output, feed_dict={model.decoder.z: zs}).reshape(zs.shape[0],
                                                                                        28, 28)
        xs[i * model.batch_size:(i + 1) * model.batch_size, :, :] = gen_x
    return xs


def sanity_check(images, std, path):
    canvas = np.ndarray((images.shape[0] * 28, images.shape[0] * 28))
    for y in range(images.shape[0]):
        sample_img_idxs = np.random.randint(low=0, high=images.shape[1], size=images.shape[0])
        for x, i in enumerate(sample_img_idxs):
            canvas[
            (images.shape[0] - y - 1) * 28:(images.shape[0] - y) * 28,
            x * 28:(x + 1) * 28
            ] = images[y, i, :, :]
    imageio.imwrite(str(path) + '/samples.png', canvas)


if __name__ == '__main__':
    args = parse_args()
    checkpoint_file = build_checkpoint_path(args.config, args.ckpt_file)
    architecture = config[args.config]

    # Load MNIST data
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Session() as session:
        # Initialize & Restore VAE
        vae = Vae(architecture)
        vae.restore(session, checkpoint_file)

        encodings, labels = get_encodings(session, vae, dataset)
        distributions = get_distributions(encodings, labels)

        save_root = RESULT_DIR / 'samples'
        if not save_root.exists():
            save_root.mkdir()
        save_sampling_path = save_root / ('sampling_%d' % args.sampling)
        if not save_sampling_path.exists():
            save_sampling_path.mkdir()

        std = 0
        n_steps = int(1 / args.step_size) + 1

        for i in range(n_steps):
            param_save_path = save_sampling_path / ('std_%.1f' % std)
            if not param_save_path.exists():
                param_save_path.mkdir()
            imgs = gen_param_results(session, vae, distributions, stdev=std, sampling=args.sampling,
                              batches=args.batches, save_path=param_save_path)
            sanity_check(imgs, std, param_save_path)
            std += args.step_size
