"""Tutorial on how to create a convolutional autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
import tensorflow as tf
import numpy as np
import math
import os
# from activations import lrelu
# from libs.utils import corrupt

print('eva cnn ctest 35000:')

# %%
def autoencoder(input_shape=[None, 36, 7, 2],#!edit! use 504 = 36*14 instead of 784
                n_filters=[2, 8, 16, 32],
                filter_sizes=[3, 3, 3, 3],
                corruption=False):
    """Build a deep denoising autoencoder w/ tied weights.

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
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training

    Raises
    ------
    ValueError
        Description
    """
    # %%
    # input to the network
    x = tf.placeholder(tf.float32, input_shape, name='x')# !edit!use float64 instead of float32
    Y = tf.placeholder(tf.float32, input_shape, name='Y')



    # %%
    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = x.get_shape().as_list()[1] #!edit! this line is false for my dataset
        y_dim = x.get_shape().as_list()[2]
        x_tensor = tf.reshape(
            x, [-1, x_dim, y_dim, n_filters[0]])# !edit!correct this line
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # %%
    # Optionally apply denoising autoencoder
    # if corruption:
    #     current_input = corrupt(current_input)

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    # first layer
    layer_i = 0
    n_output = n_filters[1]
    n_input = current_input.get_shape().as_list()[3]
    shapes.append(current_input.get_shape().as_list())
    W = tf.Variable(
        tf.random_uniform(dtype=tf.float32, shape=[
            filter_sizes[layer_i],
            filter_sizes[layer_i],
            n_input, n_output],
            minval=-1.0 / math.sqrt(n_input),
            maxval=1.0 / math.sqrt(n_input)))

    b = tf.Variable(tf.zeros(shape=[n_output], dtype=tf.float32))

    # custom padding
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    current_input = tf.pad(current_input, paddings, "SYMMETRIC")


    output = tf.nn.tanh(
        tf.add(tf.nn.conv2d(
            current_input, W, strides=[1, 1, 1, 1], padding='VALID'), b))
    current_input = output
    print('outputshape {}'.format(output.get_shape()))

    for layer_i, n_output in enumerate(n_filters[2:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform(dtype=tf.float32, shape=[
                filter_sizes[layer_i+1],
                filter_sizes[layer_i+1],
                n_input, n_output],
                minval=-1.0 / math.sqrt(n_input),
                maxval=1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros(shape=[n_output],dtype=tf.float32))
        encoder.append(W)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        current_input = tf.pad(current_input, paddings, "SYMMETRIC")
        output = tf.nn.tanh(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 1, 1, 1], padding='VALID'), b))# !edit! you should change strides
        current_input = output
        print('outputshape {}'.format(output.get_shape()))

    # %%
    # store the latent representation
    z = current_input
    n_filters.reverse()
    encoder.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the different weights
    # for layer_i, shape in enumerate(shapes[:-1]):
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
        print('outputshape {}'.format(output.get_shape()))

    layer_i = layer_i + 1
    # shape = shapes[-1]
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
    # output = lrelu(tf.add(tf.nn.conv2d_transpose(
    #     current_input, W,
    #     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
    #     strides=[1, 2, 1, 1], padding='SAME'), b))
    print('outputshape {}'.format(output.get_shape()))


    # %%
    # cost function measures pixel-wise difference
    cost = tf.reduce_mean(tf.square(tf.subtract(output,Y)))

    # %%
    return {'x': x, 'z': z, 'Y': Y, 'cost': cost, 'Y_pred':output}


# %%
def train_csi():
    """Test the convolutional autoencder using csi samples."""
    # %%
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import scipy.io as sio
    import numpy as np
    import os

    # %%
    # load csi as before
    # Data_path = "D:/faculty/final_project/H/H"
    #     # hdict = sio.loadmat(os.path.join(Data_path, 'My_perfect_H_12'))
    #     # #hdict = sio.loadmat(data_path, 'My_perfect_H_12')
    #     # h = hdict['My_perfect_H']
    hdict = sio.loadmat('drive/My Drive/Bsc_project/google_colab/My_perfect_H_12.mat')
    h = hdict['My_perfect_H']

    #uplink channel as input
    csi_ul = np.empty(shape=(35000, 36, 7, 2),dtype=np.float32)
    csi_ul_test = np.empty(shape=(5000, 36, 7, 2), dtype = np.float32)
    csi_ul[:, :, :, 0] = np.real(h[:35000, :36, :7])
    csi_ul[:, :, :, 1] = np.imag(h[:35000, :36, :7])
    csi_ul_test[:, :, :, 0] = np.real(h[35000:, :36, :7])
    csi_ul_test[:, :, :, 1] = np.imag(h[35000:, :36, :7])
    mean_csi_ul = np.mean(csi_ul, axis=0)

    #downlink channel as output
    csi_dl = np.empty(shape=(35000, 36, 7, 2), dtype=np.float32)
    csi_dl_test = np.empty(shape=(5000, 36, 7, 2), dtype=np.float32)
    csi_dl[:, :, :, 0] = np.real(h[:35000, 36:, 7:])
    csi_dl[:, :, :, 1] = np.imag(h[:35000, 36:, 7:])
    csi_dl_test[:, :, :, 0] = np.real(h[35000:, 36:, 7:])
    csi_dl_test[:, :, :, 1] = np.imag(h[35000:, 36:, 7:])
    mean_csi_dl = np.mean(csi_dl, axis=0)

    ae = autoencoder()

    # %%
    learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
    star_learning_rate = 0.0001
    op = tf.train.AdamOptimizer(learning_rate)
    optimizer = op.minimize(ae['cost'])
    # optimizer = tf.train.AdamOptimizer().minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # %%
    # Fit all training data
    #idxs = np.random.permutation(35000)
    batch_size = 4096
    n_batch = 35000 // batch_size
    n_epochs = 0

    checkpoint_dir='drive/My Drive/Colab Notebooks/test/evacnncheckpoints'
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Restoring previous model...")
        try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored")
        except:
            print("Could not restore model")
            pass
    # if os.path.isfile(checkpointfile):
    #     saver2 = tf.train.import_meta_graph(checkpointfile)
    #     saver2.restore(sess, tf.train.latest_checkpoint(os.path.join(os.getcwd(),'checkpoint')))

    for epoch_i in range(n_epochs):
        idxs = np.random.permutation(35000)
        for batch_i in range(n_batch):
            idxs_i = idxs[batch_i*batch_size: (batch_i+1)*batch_size]
            inputs = np.array([ul - mean_csi_ul for ul in csi_ul[idxs_i]])
            labels = np.array([dl - mean_csi_dl for dl in csi_dl[idxs_i]])
            sess.run(optimizer, feed_dict={ae['x']: inputs, ae['Y']: labels, learning_rate:star_learning_rate})
            calculated_cost=sess.run(ae['cost'], feed_dict={ae['x']:inputs,ae['Y']:labels, learning_rate:star_learning_rate})
            # print('batch:{}, cost:{}'.format(batch_i, calculated_cost))
            if batch_i % 100 == 99:
                log = open('drive/My Drive/Colab Notebooks/test/evacnncheckpoints/train_log.txt', 'a')
                log.write('batch: {}, cost: {}\n'.format(batch_i,calculated_cost))
                log.close()
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: inputs, ae['Y']: labels, learning_rate:star_learning_rate}))
        # print('adam learning rate: {}'.format(sess.run(op._lr_t)))
        if (epoch_i%20==1):
            # saver.save(sess, 'drive/My Drive/Colab Notebooks/test/etucnncheckpoints/my_model', global_step=0)
            saver.save(sess, checkpoint_dir + '/saved_model.ckpt')
        if epoch_i % 200 == 199:
            star_learning_rate = star_learning_rate/2

    print('test of network:\n')
    print('the average of test uplink channel is:{}'.format(np.mean(np.abs(csi_ul_test))))
    print('the average of  test downlink channel is:{}'.format(np.mean(np.abs(csi_dl_test))))
    print('and finally the loss is:{}'.format(sess.run(ae['cost'],feed_dict={ae['x']:csi_ul_test, ae['Y']:csi_dl_test, learning_rate:star_learning_rate})))
    out_images=sess.run(ae['Y_pred'], feed_dict={ae['x']:csi_ul_test, ae['Y']:csi_dl_test, learning_rate:star_learning_rate})
    out = np.empty((out_images.shape[0], out_images.shape[1], out_images.shape[2]), dtype=np.complex128)
    out = 1.0j*np.squeeze(out_images[:, :, :, 1])
    out += np.squeeze(out_images[:, :, :, 0])
    sio.savemat('drive/My Drive/Colab Notebooks/test/evacnnimages/recon_images.mat', {'h':out})
    print(r'reconstructed images saved to recon_images.mat...')



# %%
if __name__ == '__main__':
    train_csi()
