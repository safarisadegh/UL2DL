# @Author: Mohammad Sadegh Safari By Thanks to : Brandon Amos

import matplotlib.pyplot as plt
from model import BEGAN
import tensorflow as tf
import numpy as np
import math
import os
import scipy.io as sio


def load_dataset():
    hdict = sio.loadmat('../data/my_spec_ETU.mat')
    h = hdict['my_spec_ETU']
    # uplink channel as input
    csi_ul = np.empty(shape=(5000, 72, 14, 2), dtype=np.float32)
    csi_ul[:, :, :, 0] = np.real(h[35000:, :, :])
    csi_ul[:, :, :, 1] = np.imag(h[35000:, :, :])
    print(np.mean(csi_ul))
    print(np.max(csi_ul))
    print(np.max(-csi_ul))
    csi_ul = csi_ul/3.3
    return csi_ul


def save_image(generated_images, name):
    out = 1.0j * np.squeeze(generated_images[:, :, :, 1])
    out += np.squeeze(generated_images[:, :, :, 0])

    if not os.path.isdir('images'):
        os.makedirs('images')

    sio.savemat('images/' + name + '.mat', {'h': out})


def train(model, epoch_count, batch_size, z_dim, star_learning_rate, beta1, beta2, real_images):
    input_real, input_z, lrate, k_t, decay = model.model_inputs(72, 14, 2, z_dim)

    d_loss, g_loss, d_real, d_fake, generated_images, generated_images_recon, input_image, input_image_recon = model.model_loss(
        input_real, input_z, 2, z_dim, k_t, decay)

    d_opt, g_opt, d_train_opt_on_d, d_train_opt_on_g = model.model_opt(d_loss, g_loss,d_real, d_fake, lrate, beta1, beta2)



    # added for completion
    lambda_val = 0.01
    mask = tf.placeholder(tf.float32, [None] + [72, 14, 2], name='mask')
    contextual_loss = tf.reduce_sum(
        tf.contrib.layers.flatten(
            tf.abs(tf.multiply(mask, generated_images) - tf.multiply(mask, input_image))), 1)
    mse = 2*tf.reduce_mean(tf.square(tf.multiply(1 - mask, generated_images) - tf.multiply(1 - mask, input_image)))

    perceptual_loss = g_loss
    complete_loss = contextual_loss + lambda_val * perceptual_loss
    grad_complete_loss = tf.gradients(complete_loss, input_z)

    config = {}
    config['maskType'] = 'csi'
    mask_ = np.ones([72, 14, 2])
    mask_[36:, :, :] = 0
    mask_[:36, 7:, :] = 0

    batch_mask = np.resize(mask_, [batch_size, 72, 14, 2])
    zhats = np.random.uniform(-1, 1, size=(batch_size, 64))
    batch_images = load_dataset()[:batch_size, :, :, :]

    learning_rate = star_learning_rate
    iter = 0

    epoch_drop = 3

    lam = 1e-3
    gamma = .5
    k_curr = 0.0

    test_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
    checkpoint_dir = 'checkpoint'
    saver = tf.train.Saver()

    # restore previous model if there is one
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring previous model...")
            try:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored")
            except:
                print("Could not restore model")
                pass

        vel = 0
        momentum = 0.9
        for i in range(2000):

            fd = {
                input_z: zhats,
                mask: batch_mask,
                input_real: batch_images,
                lrate: learning_rate,
                k_t: 0,
                decay: 0
            }
            run = [complete_loss, grad_complete_loss, generated_images]
            loss, g, G_imgs = sess.run(run, feed_dict=fd)

            if i % 20 is 0:
                print("loss in iteration: " + str(i) + " is: " + str(np.mean(loss)))
                print('contextual loss:', sum(sess.run(contextual_loss, feed_dict=fd)) / batch_size)
                print('perceptual loss:', sess.run(perceptual_loss, feed_dict=fd))
                print('.mse: ', sess.run(mse, feed_dict=fd))
            if i % 500 is 0:
                learning_rate = learning_rate / 2

            prev_vel = np.copy(vel)
            vel = momentum * vel - learning_rate * g[0]
            zhats += -momentum * prev_vel + (1 + momentum) * vel
            zhats = np.clip(zhats, -1, 1)

            if i % 500 == 499:

                created_images = G_imgs
                save_image(created_images, "created_images"+str(i))

                masked_images = np.multiply(batch_images, batch_mask)

                inv_mask_ = 1 - mask_

                inv_batch_mask = np.resize(inv_mask_, [batch_size] + [72, 14, 2])
                inv_masked_images = np.multiply(G_imgs, inv_batch_mask)

                Recons_img = inv_masked_images + masked_images
                save_image(Recons_img, "completed_images"+str(i))


if __name__ == '__main__':

    batch_size = 1000
    z_dim = 64  # aka embedding
    learning_rate = 0.01
    beta1 = 0.5
    beta2 = 0.999
    epochs = 20

    model = BEGAN()

    with tf.Graph().as_default():
        train(model, epochs, batch_size, z_dim, learning_rate, beta1, beta2, load_dataset())
