import tensorflow as tf
import numpy as np

class LatentGan2(object):
    def __init__(self, means, stds, batch_size, decoder):
        # dropout = 0.4

        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 10, 10))


        self.decoder = decoder # Pre-trained decoder

        # self.generator = tf.keras.models.Sequential()
        # self.generator.add(tf.keras.layers.Dense(28 * 28 * 1, input_shape=(784,), kernel_initializer=tf.keras.initializers.identity))
        # self.generator.add(tf.keras.layers.Activation('relu'))
        # self.generator.add(tf.keras.layers.Dropout(dropout))
        # self.generator.add(tf.keras.layers.Dense(28 * 28 * 1))
        # self.generator.add(tf.keras.layers.Activation('relu'))
        # self.generator.add(tf.keras.layers.Dropout(dropout))
        # self.generator.add(tf.keras.layers.Flatten())
        # self.generator.add(tf.keras.layers.Reshape((28, 28, 1)))
        # self.generator.add(tf.keras.layers.Conv2D(8, 5, strides=2, padding='same',
        #                                           activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
        # self.generator.add(tf.keras.layers.Dropout(dropout))
        # self.generator.add(tf.keras.layers.UpSampling2D())
        # self.generator.add(tf.keras.layers.Conv2DTranspose(4, 5, padding='same'))
        # self.generator.add(tf.keras.layers.BatchNormalization(momentum=0.9))
        # self.generator.add(tf.keras.layers.Activation('relu'))
        # self.generator.add(tf.keras.layers.Conv2DTranspose(2, 5, padding='same'))
        # self.generator.add(tf.keras.layers.BatchNormalization(momentum=0.9))
        # self.generator.add(tf.keras.layers.Activation('relu'))
        # self.generator.add(tf.keras.layers.Conv2DTranspose(1, 5, padding='same'))
        # self.generator.add(tf.keras.layers.Activation('sigmoid'))
        # self.generator.summary()
        #
        # self.discriminator = tf.keras.models.Sequential()
        # self.discriminator.add(tf.keras.layers.Conv2D(16, 5, strides=2, input_shape=(28, 28, 1),
        #                                               padding='same',
        #                                               activation=tf.keras.layers.LeakyReLU(
        #                                                   alpha=0.2)))
        # self.discriminator.add(tf.keras.layers.Dropout(dropout))
        # self.discriminator.add(tf.keras.layers.Conv2D(32, 5, strides=2, padding='same',
        #                                               activation=tf.keras.layers.LeakyReLU(
        #                                                   alpha=0.2)))
        # self.discriminator.add(tf.keras.layers.Dropout(dropout))
        # self.discriminator.add(tf.keras.layers.Conv2D(64, 5, strides=2, padding='same',
        #                                               activation=tf.keras.layers.LeakyReLU(
        #                                                   alpha=0.2)))
        # self.discriminator.add(tf.keras.layers.Dropout(dropout))
        # self.discriminator.add(tf.keras.layers.Conv2D(128, 5, strides=1, padding='same',
        #                                               activation=tf.keras.layers.LeakyReLU(
        #                                                   alpha=0.2)))
        # self.discriminator.add(tf.keras.layers.Dropout(dropout))
        # self.discriminator.add(tf.keras.layers.Flatten())
        # self.discriminator.add(tf.keras.layers.Dense(10))
        # self.discriminator.add(tf.keras.layers.Activation('sigmoid'))
        # self.discriminator.summary()
        #
        # dm_optimizer = tf.keras.optimizers.RMSprop(lr=0.0002, decay=6e-8)
        # self.dm = tf.keras.models.Sequential()
        # self.dm.add(self.discriminator)
        # self.dm.compile(loss=tf.nn.softmax_cross_entropy_with_logits_v2, optimizer=dm_optimizer, metrics=['accuracy'])
        #
        # am_optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        # self.am = tf.keras.models.Sequential()
        # self.am.add(self.generator)
        # self.am.add(self.discriminator)
        # self.am.compile(loss=tf.nn.softmax_cross_entropy_with_logits_v2, optimizer=am_optimizer, metrics=['accuracy'])

    def train(self, sess, means, stds, batch_x, batch_y):
        # mean_x = np.matmul(batch_y, means)
        # std_x = np.matmul(batch_y, stds)
        #
        # eps = np.random.normal(0, 0.5, size=(100, 20, 3))
        # z = np.mean([mean_x + std_x * eps[:,:,i] for i in range(3)], axis=0)
        # decoded = sess.run(self.decoder.output, feed_dict={self.decoder.z: z})
        # generated = self.generator.predict(decoded)
        # x = np.concatenate((batch_x.reshape(100, 28, 28, 1), generated))
        # y = np.concatenate((batch_y, np.zeros((100, 10))))
        # d_loss = self.dm.train_on_batch(x, y)
        #
        # fake_numbers = np.random.randint(0, 10, size=100)
        # fake_labels = np.zeros(shape=(100, 10))
        # fake_labels[np.arange(100), fake_numbers] = 1
        # mean_x = np.matmul(fake_labels, means)
        # std_x = np.matmul(fake_labels, stds)
        # eps = np.random.normal(0, 0.5, size=(100, 20, 3))
        # z = np.mean([mean_x + std_x * eps[:, :, i] for i in range(3)], axis=0)
        # decoded = sess.run(self.decoder.output, feed_dict={self.decoder.z: z})
        # a_loss = self.am.train_on_batch(decoded, fake_labels)
        #
        # return d_loss, a_loss



