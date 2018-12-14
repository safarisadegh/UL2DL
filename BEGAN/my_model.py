import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import numpy as np


class BEGAN(object):
    def __init__(self, place_holder=''):
        self.place_holder = place_holder


    def gaussian_noise_layer(self, input_tensor, std=0.02):
        noise = tf.random_normal(shape=tf.shape(input_tensor), mean=0.0, stddev=std, dtype=tf.float32)

        return tf.reshape(input_tensor + noise, tf.shape(input_tensor))

    def model_inputs(self, image_width, image_height, image_channels, z_dim):
        """
        Create the model inputs/tensors
        """
        inputs_real = tf.placeholder(
            tf.float32, (None, image_width, image_height, image_channels), name='input_real')
        inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
        learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        k_t = tf.placeholder(tf.float32, name='k_t')
        decay = tf.placeholder(tf.float32, name='noise_decay')

        return inputs_real, inputs_z, learning_rate, k_t, decay

    # default aplha is 0.2, 0.01 works best for this example
    # Function from TensorFlow v1.4 for backwards compatability
    def leaky_relu(self, features, alpha=0.01, name=None):
        with ops.name_scope(name, "LeakyRelu", [features, alpha]):
            features = ops.convert_to_tensor(features, name="features")
            alpha = ops.convert_to_tensor(alpha, name="alpha")

            return math_ops.maximum(alpha * features, features)

    def fully_connected(self, x, output_shape):
        # flatten and dense
        shape = x.get_shape().as_list()
        dim = np.prod(shape[1:])

        x = tf.reshape(x, [-1, dim])
        x = tf.layers.dense(x, output_shape, activation=tf.nn.sigmoid, kernel_initializer=tf.glorot_normal_initializer())

        return x

    def decoder(self, h, n, h_dim, w_dim,  channel_dim):
        """
        Reconstruction network
        """
        #h = tf.layers.dense(h, h_dim * w_dim * n, activation=None, use_bias=False)
        h = tf.layers.dense(h, h_dim * w_dim * n, activation=None, kernel_initializer=tf.glorot_normal_initializer() )
        h = tf.reshape(h, (-1, h_dim, w_dim, n))

        conv1 = tf.layers.batch_normalization(tf.layers.conv2d(
            h, n, 3, padding="same", activation=self.leaky_relu, kernel_initializer=tf.glorot_normal_initializer()))
        conv1 = tf.layers.batch_normalization(tf.layers.conv2d(
            conv1, n, 3, padding="same", activation=self.leaky_relu, kernel_initializer=tf.glorot_normal_initializer()))

        upsample1 = tf.layers.batch_normalization(tf.image.resize_nearest_neighbor(
            conv1, size=(h_dim * 2, w_dim)))

        conv2 = tf.layers.batch_normalization(tf.layers.conv2d(
            upsample1, n, 3, padding="same", activation=self.leaky_relu, kernel_initializer=tf.glorot_normal_initializer()))
        conv2 = tf.layers.batch_normalization(tf.layers.conv2d(
            conv2, n, 3, padding="same", activation=self.leaky_relu, kernel_initializer=tf.glorot_normal_initializer()))

        upsample2 = tf.layers.batch_normalization(tf.image.resize_nearest_neighbor(
            conv2, size=(h_dim * 4, w_dim * 2)))

        conv3 = tf.layers.batch_normalization(tf.layers.conv2d(
            upsample2, n, 3, padding="same", activation=self.leaky_relu, kernel_initializer=tf.glorot_normal_initializer()))
        conv3 = tf.layers.batch_normalization(tf.layers.conv2d(
            conv3, n, 3, padding="same", activation=self.leaky_relu, kernel_initializer=tf.glorot_normal_initializer()))

        conv4 = tf.layers.conv2d(conv3, channel_dim, 3,
                                 padding="same", activation=None, kernel_initializer=tf.glorot_normal_initializer())

        return conv4

    def encoder(self, images, n, z_dim, channel_dim):
        """
        Feature extraction network
        """
        conv1 = tf.layers.batch_normalization(tf.layers.conv2d(
            images, n, 3, padding="same", activation=self.leaky_relu, kernel_initializer=tf.glorot_normal_initializer()))

        conv2 = tf.layers.batch_normalization(tf.layers.conv2d(
            conv1, n, 3, padding="same", activation=self.leaky_relu, kernel_initializer=tf.glorot_normal_initializer()))
        conv2 = tf.layers.batch_normalization(tf.layers.conv2d(
            conv2, n * 2, 3, padding="same", activation=self.leaky_relu, kernel_initializer=tf.glorot_normal_initializer()))

        subsample1 = tf.layers.batch_normalization(tf.layers.conv2d(
            conv2, n * 2, 3, strides=2, padding='same', kernel_initializer=tf.glorot_normal_initializer()))

        conv3 = tf.layers.batch_normalization(tf.layers.conv2d(subsample1, n * 2, 3,
                                 padding="same", activation=self.leaky_relu, kernel_initializer=tf.glorot_normal_initializer()))
        conv3 = tf.layers.batch_normalization(tf.layers.conv2d(
            conv3, n * 3, 3, padding="same", activation=self.leaky_relu, kernel_initializer=tf.glorot_normal_initializer()))

        subsample2 = tf.layers.batch_normalization(tf.layers.conv2d(
            conv3, n * 3, 3, strides=(2, 1), padding='same', kernel_initializer=tf.glorot_normal_initializer()))

        conv4 = tf.layers.batch_normalization(tf.layers.conv2d(subsample2, n * 3, 3,padding="same",
                                 activation=self.leaky_relu, kernel_initializer=tf.glorot_normal_initializer()))
        conv4 = tf.layers.batch_normalization(tf.layers.conv2d(
            conv4, n * 3, 3, padding="same", activation=self.leaky_relu, kernel_initializer=tf.glorot_normal_initializer()))

        h = tf.layers.batch_normalization(self.fully_connected(conv4, z_dim))

        return h

    def discriminator(self, images, z_dim, channel_dim, reuse=False):
        """
        Create the discriminator network: The autoencoder
        """
        with tf.variable_scope('discriminator', reuse=reuse):
            x = tf.layers.batch_normalization(self.encoder(images, 64, z_dim, channel_dim))
            x = self.decoder(x, 64, 18, 7, channel_dim)

            return x

    def generator(self, z, channel_dim, is_train=True):
        """
        Create the generator network: Only the encoder part
        """
        reuse = False if is_train else True
        with tf.variable_scope('generator', reuse=reuse):
            x = self.decoder(z, 64, 18, 7, channel_dim)

            return x

    def model_loss(self, input_real, input_z, channel_dim, z_dim, k_t, decay):
        """
        Get the loss for the discriminator and generator
        """
        g_model_fake = self.generator(input_z, channel_dim, is_train=True)
        noisy_input_real = self.gaussian_noise_layer(input_real, decay)

        noisy_input_fake = self.gaussian_noise_layer(g_model_fake, decay)
        d_model_real = self.discriminator(noisy_input_real, z_dim, channel_dim)
        d_model_fake = self.discriminator(
            noisy_input_fake, z_dim, channel_dim, reuse=True)

        print("---------------------")
        print(input_z.shape)
        print(g_model_fake.shape)
        
        print("---------------------")
        print(input_real.shape)
        print(d_model_real.shape)
        print(d_model_fake.shape)
        print(channel_dim)
        print("---------------------")
        # # l1 loss
        # d_real = tf.reduce_mean(tf.abs(input_real - d_model_real))
        # d_fake = tf.reduce_mean(tf.abs(g_model_fake - d_model_fake))
        # l1 loss
        d_real = tf.sqrt(tf.reduce_mean(tf.square(noisy_input_real - d_model_real)))
        d_fake = tf.sqrt(tf.reduce_mean(tf.square(noisy_input_fake - d_model_fake)))

        d_loss = d_real - k_t * d_fake
        g_loss = d_fake

        return d_loss, g_loss, d_real, d_fake, g_model_fake, d_model_fake, input_real, d_model_real

    def model_opt(self, d_loss, g_loss, d_real, d_fake, learning_rate, beta1, beta2=0.999):
        """
        Get optimization operations
        """
        # Get variables
        g_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, "generator")
        d_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, "discriminator")

        # Optimize
        d_train_opt = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1, beta2=beta2).minimize(d_loss, var_list=d_vars)

        d_train_opt_on_d = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1, beta2=beta2).minimize(d_real, var_list=d_vars)

        d_train_opt_on_g = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1, beta2=beta2).minimize(d_real-d_fake, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1, beta2=beta2).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt,d_train_opt_on_d, d_train_opt_on_g
