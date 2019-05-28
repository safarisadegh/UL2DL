"""@Author: Moahammad Sadegh Safari by thanks to @Parag K. Mital
"""
import tensorflow as tf
import numpy as np
import math
import os
import scipy.io as sio


def autoencoder(input_shape=[None, 36, 7, 2],
                n_filters=[2, 8, 16, 32],
                filter_sizes=[3, 3, 3, 3]):
    """Build a deep neural net to predict DL CSI.

    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training


    """

    # input to the network
    x = tf.placeholder(tf.float32, input_shape, name='x')
    Y = tf.placeholder(tf.float32, input_shape, name='Y')



    # %%
    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = x.get_shape().as_list()[1]
        y_dim = x.get_shape().as_list()[2]
        x_tensor = tf.reshape(
            x, [-1, x_dim, y_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # Build the network

    # first layer
    layer_i = 0
    n_output = n_filters[1]
    n_input = current_input.get_shape().as_list()[3]
    W = tf.Variable(
        tf.random_uniform(dtype=tf.float32, shape=[
            filter_sizes[layer_i],
            filter_sizes[layer_i],
            n_input, n_output],
            minval=-1.0 / math.sqrt(n_input),
            maxval=1.0 / math.sqrt(n_input)))

    b = tf.Variable(tf.zeros(shape=[n_output], dtype=tf.float32))

    # symmetric padding for first layer
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    current_input = tf.pad(current_input, paddings, "SYMMETRIC")

    output = tf.nn.tanh(
        tf.add(tf.nn.conv2d(
            current_input, W, strides=[1, 1, 1, 1], padding='VALID'), b))
    current_input = output

    for layer_i, n_output in enumerate(n_filters[2:]):
        n_input = current_input.get_shape().as_list()[3]
        W = tf.Variable(
            tf.random_uniform(dtype=tf.float32, shape=[
                filter_sizes[layer_i+1],
                filter_sizes[layer_i+1],
                n_input, n_output],
                minval=-1.0 / math.sqrt(n_input),
                maxval=1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros(shape=[n_output],dtype=tf.float32))

        # symmetric padding for second layer
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        current_input = tf.pad(current_input, paddings, "SYMMETRIC")
        output = tf.nn.tanh(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 1, 1, 1], padding='VALID'), b))
        current_input = output

    n_filters.reverse()

    for layer_i in range(2):
        W = tf.Variable(
            tf.random_uniform(dtype=tf.float32, shape=[
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_filters[layer_i], n_filters[layer_i + 1]],
                              minval=-1.0 / math.sqrt(n_filters[layer_i]),
                              maxval=1.0 / math.sqrt(n_filters[layer_i])))
        b = tf.Variable(tf.zeros(shape=[n_filters[layer_i + 1]], dtype=tf.float32))
        output = tf.nn.tanh(tf.add(
            tf.nn.conv2d(
                current_input, W,
                strides=[1, 1, 1, 1], padding='SAME'), b))
        current_input = output

    layer_i = layer_i + 1
    W = tf.Variable(
        tf.random_uniform(dtype=tf.float32, shape=[
            filter_sizes[layer_i],
            filter_sizes[layer_i],
            n_filters[layer_i], n_filters[layer_i + 1]],
                          minval=-1.0 / math.sqrt(n_filters[layer_i]),
                          maxval=1.0 / math.sqrt(n_filters[layer_i])))
    b = tf.Variable(tf.zeros(shape=[n_filters[layer_i + 1]], dtype=tf.float32))
    output = tf.add(
        tf.nn.conv2d(
            current_input, W,
            strides=[1, 1, 1, 1], padding='SAME'), b)

    # MSE cost function for prediction
    cost = tf.reduce_mean(tf.square(tf.subtract(output,Y)))

    # %%
    return {'x': x, 'Y': Y, 'cost': cost, 'Y_pred':output}


# %%
def train_csi():
    """Test the convolutional network using csi samples."""
    # %%
    # load CSI dataset with 40K samples
    hdict = sio.loadmat('../data/My_perfect_H_12.mat')
    h = hdict['My_perfect_H']

    # uplink channel as input
    csi_ul = np.empty(shape=(35000, 36, 7, 2), dtype=np.float32)
    csi_ul_test = np.empty(shape=(4000, 36, 7, 2), dtype=np.float32)
    csi_ul_val = np.empty(shape=(1000, 36, 7, 2), dtype=np.float32)
    csi_ul[:, :, :, 0] = np.real(h[:35000, :36, :7])
    csi_ul[:, :, :, 1] = np.imag(h[:35000, :36, :7])
    csi_ul_val[:, :, :, 0] = np.real(h[35000:36000, :36, :7])
    csi_ul_val[:, :, :, 1] = np.imag(h[35000:36000, :36, :7])
    csi_ul_test[:, :, :, 0] = np.real(h[36000:, :36, :7])
    csi_ul_test[:, :, :, 1] = np.imag(h[36000:, :36, :7])
    mean_csi_ul = np.mean(csi_ul, axis=0)

    # downlink channel as output
    csi_dl = np.empty(shape=(35000, 36, 7, 2), dtype=np.float32)
    csi_dl_test = np.empty(shape=(4000, 36, 7, 2), dtype=np.float32)
    csi_dl_val = np.empty(shape=(1000, 36, 7, 2), dtype=np.float32)
    csi_dl[:, :, :, 0] = np.real(h[:35000, 36:, 7:])
    csi_dl[:, :, :, 1] = np.imag(h[:35000, 36:, 7:])
    csi_dl_val[:, :, :, 0] = np.real(h[35000:36000, 36:, 7:])
    csi_dl_val[:, :, :, 1] = np.imag(h[35000:36000, 36:, 7:])
    csi_dl_test[:, :, :, 0] = np.real(h[36000:, 36:, 7:])
    csi_dl_test[:, :, :, 1] = np.imag(h[36000:, 36:, 7:])
    mean_csi_dl = np.mean(csi_dl, axis=0)

    ae = autoencoder()

    val_input = np.array([ul - mean_csi_ul for ul in csi_ul_val])
    val_output = np.array([dl - mean_csi_dl for dl in csi_dl_val])
    csi_ul_test = np.array([ul - mean_csi_ul for ul in csi_ul_test])
    csi_dl_test = np.array([dl - mean_csi_dl for dl in csi_dl_test])

    # defining a placeholder for learning rate
    learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
    star_learning_rate = 0.001
    op = tf.train.AdamOptimizer(learning_rate)
    optimizer = op.minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    batch_size = 32
    n_batch = 35000 // batch_size
    n_epochs = 100

    checkpoint_dir = 'checkpoint'
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Restoring previous model...")
        try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored")
        except:
            print("Could not restore model")
            pass

    for epoch_i in range(n_epochs):
        idxs = np.random.permutation(35000)
        for batch_i in range(n_batch):
            idxs_i = idxs[batch_i*batch_size: (batch_i+1)*batch_size]

            # normalize dl and ul csi's
            inputs = np.array([ul - mean_csi_ul for ul in csi_ul[idxs_i]])
            labels = np.array([dl - mean_csi_dl for dl in csi_dl[idxs_i]])
            sess.run(optimizer, feed_dict={ae['x']: inputs, ae['Y']: labels, learning_rate: star_learning_rate})

        train_loss = sess.run(2*ae['cost'], feed_dict={ae['x']: inputs, ae['Y']: labels})
        val_loss = sess.run(2*ae['cost'], feed_dict={ae['x']: val_input, ae['Y']: val_output})

        print('epoch: ', epoch_i, ", train_loss: ", train_loss, ", val_loss: ", val_loss)
        if epoch_i % 5 == 1:
            saver.save(sess, checkpoint_dir + '/saved_model.ckpt')
        if epoch_i % 40 == 39:
            star_learning_rate = star_learning_rate/2

    print('and finally the loss is:{}'.format(2*sess.run(ae['cost'],
                                                         feed_dict={ae['x']: csi_ul_test, ae['Y']: csi_dl_test})))
    out_images = sess.run(ae['Y_pred'], feed_dict={ae['x']: csi_ul_test})
    out_images = np.array([dl + mean_csi_dl for dl in out_images])
    out = 1.0j*np.squeeze(out_images[:, :, :, 1])
    out += np.squeeze(out_images[:, :, :, 0])

    inp_images = np.array([ul + mean_csi_ul for ul in csi_ul_test])
    inp = 1.0j*np.squeeze(inp_images[:, :, :, 1])
    inp += np.squeeze(inp_images[:, :, :, 0])

    ground_truth_images = np.array([dl + mean_csi_dl for dl in csi_dl_test])
    ground_truth = 1.0j*np.squeeze(ground_truth_images[:, :, :, 1])
    ground_truth += np.squeeze(ground_truth_images[:, :, :, 0])

    if not os.path.isdir('images'):
        os.makedirs('images')
    sio.savemat('images/recon_images.mat', {'prediction': out, 'input': inp, 'ground_truth': ground_truth})
    print(r'reconstructed images saved to images/recon_images.mat...')


if __name__ == '__main__':
    train_csi()
