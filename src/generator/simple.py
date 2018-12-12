import numpy as np


class SimpleGenerator(object):
    def __init__(self, batch_size, decoder):
        self.batch_size = batch_size
        self.decoder = decoder
        self.eps_stdev = 0.5
        self.samping = 1

    def generate_batches(self, sess, dist, stdev=0.5, sampling=2, batches=1):
        xs = np.empty((self.batch_size * batches, 28, 28))
        for i in range(batches):
            eps = np.random.normal(scale=stdev,
                                   size=(self.batch_size, dist.shape[0], sampling))
            eps_mean = np.mean(eps, axis=2)
            z_mean = dist[:, 0]
            z_stdv = dist[:, 1]
            zs = z_mean + eps_mean * z_stdv
            gen_x = sess.run(self.decoder.output, feed_dict={self.decoder.input: zs}).reshape(
                zs.shape[0],
                28, 28)
            xs[i * self.batch_size:(i + 1) * self.batch_size, :, :] = gen_x
        return xs

    def generate(self, sess, stdev=0.5, sampling=2, batches=1):
        results = np.empty((self.distributions.shape[0], self.batch_size * batches, 28, 28),
                           dtype=np.float32)
        for y in range(self.distributions.shape[0]):
            # save_to = save_path / ('label_%d' % y)
            # if not save_to.exists():
            # save_to.mkdir()
            dist = self.distributions[y]
            imgs = self.generate_batches(sess, dist, stdev, sampling, batches)
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

    def train(self, sess, encodings, labels, classifier, step_size=0.1, sampling=[1, 2, 3, 5],
              batches=1):
        self.set_distributions(encodings, labels)
        num_steps = int(1 / step_size) + 1

        table = np.empty((len(sampling), num_steps, 5))
        for i, s in enumerate(sampling):
            eps_stdev = 0
            for j in range(num_steps):
                results = self.generate(sess, eps_stdev, s, batches)
                avg_acc = 0.
                avg_cos_sim = 0.
                for y in range(results.shape[0]):
                    lab_results = results[y, :, :, :].reshape(-1, 784)
                    lab_labels = np.zeros((lab_results.shape[0], results.shape[0]))
                    lab_labels[:, y] = 1.
                    cos_sim = self.cos_sim(lab_results, sampling=100)
                    avg_cos_sim += cos_sim / results.shape[0]
                    _, acc = classifier.evaluate(lab_results.reshape(-1, 28, 28, 1), lab_labels,
                                                 verbose=0)
                    avg_acc += acc / results.shape[0]
                score = avg_cos_sim + 1e1 * (1 - avg_acc)
                print('%d %.1f %.16f %.16f %.16f' % (s, eps_stdev, avg_cos_sim, avg_acc, score))
                table[i, j, :] = np.array([s, eps_stdev, avg_cos_sim, avg_acc, score])
                eps_stdev += step_size

        return table
