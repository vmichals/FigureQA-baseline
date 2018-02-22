# coding: utf-8

import numpy as np
import tensorflow as tf


def get_conv1_filter_grid_img(conv1_w, pad=1):
    """Creates an grid of convnet filters

    Args:
        conv1_w (tf.Tensor): The conv net kernel tensor.
        pad (int): how much padding around grid cells

    Returns:
        tf.Tensor: A grid of convnet filters
    """

    h, w, nchannels, b = conv1_w.get_shape().as_list()
    grid_w = np.int32(np.ceil(np.sqrt(np.float32(b))))
    grid_h = grid_w
    v_min = tf.reduce_min(conv1_w)
    v_max = tf.reduce_max(conv1_w)
    conv1_w = (conv1_w - v_min) / (v_max - v_min)

    conv1_w = tf.pad(
        conv1_w, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]),
        mode='CONSTANT'
    )

    H = h + 2 * pad
    W = w + 2 * pad

    conv1_w = tf.transpose(conv1_w, (3, 0, 1, 2))
    # pad to get a square number of grid cells
    conv1_w = tf.pad(
        conv1_w, tf.constant([[0, grid_w*grid_h - b], [0, 0], [0, 0], [0, 0]]),
                             mode='CONSTANT'
    )

    conv1_w = tf.reshape(
        conv1_w,
        tf.stack([grid_w, H * grid_h, W, nchannels])
    )

    conv1_w = tf.transpose(conv1_w, (0, 2, 1, 3))

    conv1_w = tf.reshape(
        conv1_w,
        tf.stack([1, W * grid_w, H * grid_h, nchannels])
    )

    conv1_w = tf.transpose(conv1_w, (2, 1, 3, 0))


    conv1_w = tf.transpose(conv1_w, (3, 0, 1, 2))

    return conv1_w



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    with tf.Session() as sess:
        w = tf.get_variable('w', shape=(16, 16, 3, 23), dtype=np.float32)
        sess.run(tf.global_variables_initializer())
        w_grid_img = get_conv1_filter_grid_img(w, pad=1)

        w_grid_img_np = sess.run(w_grid_img)
        plt.imshow(w_grid_img_np[0])
        #for i in range(w_grid_img_np.shape[0]):
        #    n = int(np.ceil(np.sqrt(w_grid_img_np.shape[0])))
        #    plt.subplot(n, n, i+1)
        #    plt.imshow(w_grid_img_np[i])
        plt.show()


