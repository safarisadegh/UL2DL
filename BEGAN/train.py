# @Author: Mohammad Sadegh Safari By Thanks to : @JorgeCeja

from model import BEGAN
import tensorflow as tf
import numpy as np
import math
import os
import scipy.io as sio


def load_dataset():
    hdict = sio.loadmat('../data/my_spec_ETU.mat')

    h = hdict['my_spec_ETU']

    # load 35K of 72*14 frames as input of BEGAN
    csi_ul = np.empty(shape=(35000, 72, 14, 2), dtype=np.float32)
    csi_ul[:, :, :, 0] = np.real(h[:35000, :, :])
    csi_ul[:, :, :, 1] = np.imag(h[:35000, :, :])
    print(np.mean(csi_ul))
    print(np.max(csi_ul))
    print(np.max(-csi_ul))
    # 3.3 is maximum value of real and imaginary parts
    csi_ul = csi_ul / 3.3
    return csi_ul


def save_image(Test_result, epoch, batch_number):
    # generated_images
    tmp_image = Test_result[0]
    out = 1.0j * np.squeeze(tmp_image[:, :, :, 1])
    out += np.squeeze(tmp_image[:, :, :, 0])
    generated_images = out

    # reconstruction of generated_images
    tmp_image = Test_result[1]
    out = 1.0j * np.squeeze(tmp_image[:, :, :, 1])
    out += np.squeeze(tmp_image[:, :, :, 0])
    generated_images_recon = out

    # real image
    tmp_image = Test_result[2]
    out = 1.0j * np.squeeze(tmp_image[:, :, :, 1])
    out += np.squeeze(tmp_image[:, :, :, 0])
    Input_image = out

    # reconstruction of real image
    tmp_image = Test_result[3]
    out = 1.0j * np.squeeze(tmp_image[:, :, :, 1])
    out += np.squeeze(tmp_image[:, :, :, 0])
    Input_image_recon = out

    if not os.path.isdir('images'):
        os.makedirs('images')

    sio.savemat('images/Results_' + str(epoch + 1) +
                '_batch' + str(batch_number + 1) + '.mat', {'etu_generated_images': generated_images,
                                                            'etu_generated_images_recon': generated_images_recon,
                                                            'etu_Input_image': Input_image,
                                                            'etu_Input_image_recon': Input_image_recon})


def train(model, epoch_count, batch_size, z_dim, star_learning_rate, beta1, beta2, real_images):
    input_real, input_z, lrate, k_t, decay = model.model_inputs(72, 14, 2, z_dim)

    d_loss, g_loss, d_real, d_fake, generated_images, generated_images_recon, input_image, input_image_recon = model.model_loss(
        input_real, input_z, 2, z_dim, k_t, decay)

    d_opt, g_opt, d_train_opt_on_d, d_train_opt_on_g = model.model_opt(d_loss, g_loss, d_real, d_fake, lrate, beta1,
                                                                       beta2)

    losses = []
    learning_rate = 0
    iter = 0

    epoch_drop = 3
    noise_decay=0.2

    lam = 1e-2
    gamma = .7
    k_curr = 0.0

    checkpoint_dir = 'checkpoint'
    saver = tf.train.Saver()

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # restore previous model if there is one
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring previous model...")
            try:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored")
            except:
                print("Could not restore model")
                pass

        for epoch_i in range(epoch_count):
            idxs = np.random.permutation(35000)

            learning_rate = star_learning_rate * \
                            math.pow(0.2, math.floor((epoch_i + 1) / epoch_drop))

            for batch_i in range(35000 // batch_size):
                iter += 1
                idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
                batch_images = np.array([ul for ul in real_images[idxs_i]])

                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

                _, d_real_curr = sess.run([d_opt, d_real], feed_dict={
                    input_z: batch_z, input_real: batch_images,
                    lrate: learning_rate, k_t: k_curr, decay: noise_decay / (epoch_i + 1)})

                _, d_fake_curr, disc_loss = sess.run([g_opt, d_fake, d_loss], feed_dict={
                    input_z: batch_z, input_real: batch_images,
                    lrate: learning_rate, k_t: k_curr, decay: noise_decay / (epoch_i + 1)})

                k_curr = k_curr + lam * (gamma * d_real_curr - d_fake_curr)
                if k_curr < 0:
                    _, d_fake_curr, disc_loss = sess.run([g_opt, d_fake, d_loss], feed_dict={
                        input_z: batch_z, input_real: batch_images,
                        lrate: learning_rate, k_t: k_curr, decay: noise_decay /(epoch_i + 1)})
                    k_curr = 0.0

                # save convergence measure
                if batch_i % 100 == 0:
                    measure = d_real_curr + \
                              np.abs(gamma * d_real_curr - d_fake_curr)
                    losses.append(measure)

                    print("nEpoch {}/{}...".format(epoch_i + 1, epoch_count), ' batch :', batch_i,
                          ', Convergence measure: {:.4}'.format(measure),
                          ', d_loss[{:.4}], g_loss[{:.4}], k[{:.4}]'.format(disc_loss, d_fake_curr, k_curr))

                # save test and batch images
                if iter % 500 == 1:
                    Test_result = sess.run([generated_images, generated_images_recon, input_image, input_image_recon],
                                           feed_dict={
                                               input_z: batch_z, input_real: batch_images, decay: noise_decay / (epoch_i + 1)})

                    save_image(Test_result, epoch_i, batch_i)

                    saver.save(sess, checkpoint_dir + '/saved_model.ckpt')

        print('Training steps: ', iter)


if __name__ == '__main__':
    batch_size = 32
    z_dim = 64  # aka embedding
    learning_rate = 2e-5
    beta1 = 0.5
    beta2 = 0.999
    epochs = 20

    model = BEGAN()

    with tf.Graph().as_default():
        train(model, epochs, batch_size, z_dim, learning_rate, beta1, beta2, load_dataset())
