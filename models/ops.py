#!/usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf


def conv2d(inputs, num_channels, is_training, activation_fn=tf.nn.relu,
           k_h=5, k_w=5, s_h=1, s_w=1,
           weights_initializer=None,
           biases_initializer=tf.zeros_initializer(), padding='SAME',
           use_batch_norm=False, use_dropout=False, keep_prob=.5,
           name='conv2d'):
    """A convolutional layer with optional batch norm and dropout

    Args:
        inputs (tf.Tensor): layer input
        num_channels (int): number of output channels
        is_training (tf.bool): whether we're in training mode
        activation_fn (callable): the activation function (pass None for linear
            activations)
        k_h (int): kernel height
        k_w (int): kernel width
        s_h (int): vertical stride
        s_w (int): horizontal stride
        weights_initializer: an initializer for the weights
        biases_initializer: an initializer for the biases
        padding (str): the type of padding ("SAME" or "VALID")
        use_batch_norm (bool): whether to apply batch norm to this layer
        use_dropout (bool): whether to apply dropout to this layer
        keep_prob (float): the probability of keeping a unit in dropout
        name (str): the name of the layer

    Returns:
        tf.Tensor: the layer output
    """
    with tf.variable_scope(name):
        weights = tf.get_variable(
            'weights', shape=(k_h, k_w, inputs.get_shape()[-1], num_channels),
            initializer=weights_initializer)
        biases = tf.get_variable('biases', (num_channels,),
                                 initializer=biases_initializer)
        conv = tf.nn.conv2d(inputs, filter=weights,
                            strides=(1, s_h, s_w, 1), padding=padding)
        if use_dropout:
            conv = tf.contrib.layers.dropout(
                conv, keep_prob=keep_prob, is_training=is_training
            )
        if activation_fn is None:
            activation_fn = lambda x: x
        activation = activation_fn(conv + biases)
        if use_batch_norm:
            return tf.contrib.layers.batch_norm(
                activation, center=True, scale=True, decay=.9,
                is_training=is_training, updates_collections=None)
        else:
            return activation

def fc(inputs, num_outputs, is_training, activation_fn=tf.nn.relu,
       use_batch_norm=False, use_dropout=False, keep_prob=.5, name='fc',
       **kwargs):
    """A fully-connected layer with optional batch norm and dropout

    Args:
        inputs (tf.Tensor): layer input
        num_channels (int): number of output channels
        is_training (tf.bool): whether we're in training mode
        activation_fn (callable): the activation function (pass None for linear
            activations)
        use_batch_norm (bool): whether to apply batch norm to this layer
        use_dropout (bool): whether to apply dropout to this layer
        keep_prob (float): the probability of keeping a unit in dropout
        name (str): the name of the layer
        kwargs: keyword arguments passed through to
            tf.contrib.layers.fully_connected

    Returns:
        tf.Tensor: the layer output
    """
    with tf.variable_scope(name):
        h = tf.contrib.layers.fully_connected(
            inputs, num_outputs=num_outputs, activation_fn=None,
            **kwargs)
        if use_dropout:
            h = tf.contrib.layers.dropout(
                h, keep_prob=keep_prob, is_training=is_training
            )
        if activation_fn is None:
            activation_fn = lambda x: x
        h = activation_fn(h)
        if use_batch_norm:
            return tf.contrib.layers.batch_norm(
                h, center=True, scale=True, decay=.9, is_training=is_training,
                updates_collections=None)
        else:
            return h
# vim: set ts=4 sw=4 sts=4 expandtab:
