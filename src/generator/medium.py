import numpy as np
import itertools
import time


class MediumGenerator(object):
    def __init__(self, batch_size, decoder):
        self.batch_size = batch_size
        self.decoder = decoder
        self.eps_stdev = 0.5
        self.samping = 1

    def generate_batches(self, sess, dist, stdevs, sampling=2):
        eps = np.empty((self.batch_size, dist.shape[0], sampling))
        for i in range(stdevs.shape[0]):
            eps[:, i, :] = np.random.normal(scale=stdevs[i],
                                            size=(self.batch_size, sampling))
        eps_mean = np.mean(eps, axis=2)
        z_mean = dist[:, 0]
        z_stdv = dist[:, 1]
        zs = z_mean + eps_mean * z_stdv
        gen_x = sess.run(self.decoder.output, feed_dict={self.decoder.input: zs}).reshape(
            zs.shape[0],
            28, 28)
        return gen_x

    def generate(self, sess, stdevs, sampling=2):
        results = np.empty((self.distributions.shape[0], self.batch_size, 28, 28),
                           dtype=np.float32)
        for y in range(self.distributions.shape[0]):
            # save_to = save_path / ('label_%d' % y)
            # if not save_to.exists():
            # save_to.mkdir()
            dist = self.distributions[y]
            imgs = self.generate_batches(sess, dist, stdevs, sampling)
            results[y, :, :, :] = imgs
            # np.save(save_to / 'imgs', imgs)
        return results

    def generate2(self, sess, stdevs, sampling=2):
        results = np.empty((self.distributions.shape[0], self.batch_size, 28, 28),
                           dtype=np.float32)
        for y in range(self.distributions.shape[0]):
            # save_to = save_path / ('label_%d' % y)
            # if not save_to.exists():
            # save_to.mkdir()
            stdevs_y = stdevs[y, :]
            dist = self.distributions[y]
            imgs = self.generate_batches(sess, dist, stdevs_y, sampling)
            results[y, :, :, :] = imgs
            # np.save(save_to / 'imgs', imgs)
        return results

    def set_distributions(self, encodings, labels):
        self.distributions = np.ndarray((labels.shape[1], encodings.shape[1], 2))
        for y in range(labels.shape[1]):
            idxs = np.where(labels[:, y] == 1)[0]
            encodings_cond_y = np.take(encodings, idxs, axis=0)
            for dim in range(encodings.shape[1]):
                feature_values = encodings_cond_y[:, dim]
                self.distributions[y, dim, 0] = np.mean(feature_values)
                self.distributions[y, dim, 1] = np.std(feature_values)

    def cos_sim(self, images, sampling):
        score = 0.
        for i in range(sampling):
            x_idx = y_idx = 0
            while x_idx == y_idx:
                x_idx = np.random.randint(0, images.shape[0])
                y_idx = np.random.randint(0, images.shape[0])
            x = images[x_idx, :]
            y = images[y_idx, :]
            score += np.dot(x, y) / (np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2))
        return score / sampling

    def train(self, sess, encodings, labels, classifier):
        # Medium-1
        self.set_distributions(encodings, labels)
        opt_score = 1e10
        opt_val = None

        vals = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        combinations = itertools.combinations_with_replacement(vals, 10)
        for combination in combinations:
            eps_stdevs = np.array(combination)
            results = self.generate(sess, eps_stdevs, 1)
            avg_acc = 0.
            avg_cos_sim = 0.
            for y in range(results.shape[0]):
                lab_results = results[y, :, :, :].reshape(-1, 784)
                lab_labels = np.zeros((lab_results.shape[0], results.shape[0]))
                lab_labels[:, y] = 1.
                cos_sim = self.cos_sim(lab_results, sampling=10)
                avg_cos_sim += cos_sim / results.shape[0]
                _, acc = classifier.evaluate(lab_results.reshape(-1, 28, 28, 1), lab_labels,
                                             verbose=0)
                avg_acc += acc / results.shape[0]
            score = avg_cos_sim + 1e1 * (1 - avg_acc)
            if score < opt_score:
                opt_score = score
                opt_val = eps_stdevs
                print(eps_stdevs)
                print('%.16f %.16f %.16f' % (avg_cos_sim, avg_acc, score))

        return opt_score, opt_val

    def train2(self, sess, encodings, labels, classifier):
        # Medium-2
        self.set_distributions(encodings, labels)
        opt_score = 1e10 * np.ones(10)
        opt_acc = np.zeros(10)
        opt_cos = np.ones(10)
        opt_val = np.empty((10, 10))

        vals = [0., 0.1, 0.2, 0.3]
        for val in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
            vals.append(val)
            combinations = itertools.combinations_with_replacement(vals, 9)
            for combination in combinations:
                if val in combination:
                    for val_last in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
                        eps_stdevs = np.array(combination + tuple((val_last,)))
                        results = self.generate(sess, eps_stdevs, 1)
                        for y in range(results.shape[0]):
                            lab_results = results[y, :, :, :].reshape(-1, 784)
                            lab_labels = np.zeros((lab_results.shape[0], results.shape[0]))
                            lab_labels[:, y] = 1.
                            cos_sim = self.cos_sim(lab_results, sampling=10)
                            _, acc = classifier.evaluate(lab_results.reshape(-1, 28, 28, 1), lab_labels,
                                                         verbose=0)
                            score = cos_sim + 1e1 * (1 - acc)
                            if score < opt_score[y]:
                                opt_score[y] = score
                                opt_cos[y] = cos_sim
                                opt_acc[y] = acc
                                opt_val[y, :] = eps_stdevs
                                for y in range(results.shape[0]):
                                    print(y, opt_val[y], '%.16f' % (opt_score[y]))
                                print(np.mean(opt_score), np.mean(opt_cos), np.mean(opt_acc))
                                print(time.time())
            for y in range(10):
                print(y, opt_val[y], '%.16f' % (opt_score[y]))
            print(np.mean(opt_score), np.mean(opt_cos), np.mean(opt_acc))
            print(time.time())

        return opt_score, opt_val