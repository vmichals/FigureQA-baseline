#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import division

from functools import reduce
from operator import mul

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops import embedding_ops, rnn
from tensorflow.python.framework.ops import IndexedSlices

from .ops import conv2d, fc
from util.decorators import define_scope
from util.plot_tools import get_conv1_filter_grid_img


class CNNBaselineModel(object):
    """CNN-LSTM VQA Baseline

    Implements a basic VQA model, that feeds the concatenation of a CNN image
    embedding and a LSTM question embedding to a MLP classifier predicting the
    answer.

    Args:
        is_training (bool): indicating whether mode is in training mode
        config (dict): model configuration
        image_pad_size (tuple of int): (height, width), specifying to what size
            the image should be padded (used for random cropping data
            augmentation)
        dictionary_size (int): size of the question vocabulary, defaults to
            config['question_encoder']['dictionary_size'] if it exists.

    Raises:
        KeyError: if dictionary_size is None and the config dictionary does not
            have the corresponding entry.
    """

    def __init__(self, is_training, config,
                 image_pad_size=None, dictionary_size=None):
        self._is_training = is_training
        self._config = config
        self._image_pad_size = image_pad_size
        if dictionary_size is None:
            dictionary_size = config['question_encoder']['dictionary_size']
        self._dictionary_size = dictionary_size

    @define_scope
    def _image_encoder(self, img):
        """Computes an embedding of the input image

        Args:
            img (tf.Tensor): a batch of input images (shape BxHxWxC, where
                B is the batch size and H, W, C are height, width and number of
                channels).

        Returns:
            tf.Tensor: the image embedding

        Raises:
            ValueError: if the image encoder type specified in the config is
                unsupported.
        """
        h, w, c = img.get_shape().as_list()[1:]

        if self._image_pad_size is not None:
            # pad
            pad_h = (self._image_pad_size[0] - img.get_shape()[1]) // 2
            pad_w = (self._image_pad_size[1] - img.get_shape()[2]) // 2
            pad_img = tf.pad(img, (0, pad_h, pad_w, 0))

            # pad and crop only if is_training is True
            img = tf.cond(self._is_training,
                          true_fn=lambda: tf.random_crop(
                              pad_img, (tf.shape(img)[0], h, w, c)),
                          false_fn=img,
                          strict=True)

        cfg = self._config['image_encoder']
        if cfg['type'] == 'conv-only':
            assert img.shape.ndims == 4
            for l, layer_cfg in enumerate(cfg['params']):
                layer_name = 'conv{}'.format(l)
                img = conv2d(
                    img, is_training=self._is_training,
                    weights_initializer=tf.truncated_normal_initializer(
                        stddev=0.1),
                    biases_initializer=tf.constant_initializer(.1),
                    name=layer_name, **layer_cfg)
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    if l == 0:
                        tf.summary.image(
                            layer_name,
                            get_conv1_filter_grid_img(
                                tf.get_variable('{}/weights'.format(layer_name))
                            ),
                            collections=['visualizations']
                        )
                    tf.summary.histogram(
                        layer_name,
                        tf.get_variable('{}/weights'.format(layer_name)),
                        collections=['debug']
                    )

        elif cfg['type'] == 'conv-fc':
            assert img.shape.ndims == 4
            for l, layer_cfg in enumerate(cfg['params']):
                layer_name = 'conv{}'.format(l)
                img = conv2d(img, is_training=self._is_training,
                             name=layer_name, **layer_cfg)
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    if l == 0:
                        tf.summary.image(
                            layer_name,
                            get_conv1_filter_grid_img(
                                tf.get_variable('{}/weights'.format(layer_name))
                            ),
                            collections=['visualizations']
                        )
                    tf.summary.histogram(
                        layer_name,
                        tf.get_variable('{}/weights'.format(layer_name)),
                        collections=['debug']
                    )
            img = tf.reshape(
                img, (-1, reduce(mul, img.get_shape().as_list()[1:])))
            params = cfg['fc_params']
            for l, layer_cfg in enumerate(params):
                layer_name = 'fc{}'.format(l)
                img = fc(img, is_training=self._is_training,
                         name=layer_name, **layer_cfg)
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    tf.summary.histogram(
                        layer_name,
                        tf.get_variable('{}/fully_connected/weights'.format(
                            layer_name)),
                        collections=['debug']
                    )
        else:
            raise ValueError('unsupported image encoder type specified')

        return img

    @define_scope
    def _question_encoder(self, q, qlen):
        """Computes an embedding of the input question

        Args:
            q (tf.Tensor): a batch of input questions (shape BxQ, where
                B is the batch size and Q is the maximum length of questions in
                the batch).
            qlen (tf.Tensor): a batch of input question lengths (shape B, where
                B is the batch size)

        Returns:
            tf.Tensor: the question embedding

        Raises:
            ValueError: if the question encoder type specified in the config is
                unsupported.
        """
        cfg = self._config['question_encoder']
        params = cfg['params']

        if cfg['type'] == 'lstm':
            if 'embedding_size' in cfg and cfg['embedding_size'] is not None:
                embeddings = tf.get_variable(
                    'embeddings', shape=(self._dictionary_size,
                                         cfg['embedding_size'])
                )
                q = embedding_ops.embedding_lookup(embeddings, q)

            lstm_cell = LSTMCell(params['num_units'])
            # TODO: be careful about order of axes (time-major or not?)
            # q should be the last lstm state (
            _, q = rnn.dynamic_rnn(lstm_cell, q, dtype='float32',
                                   sequence_length=qlen,
                                   time_major=False)

            q = tf.concat(q, axis=1)

            assert q.shape.ndims == 2

        else:
            raise ValueError('unsupported question encoder type specified')
        return q

    @define_scope
    def _mlp(self, x):
        """Computes the prediction with a MLP classifier

        Args:
            x (tf.Tensor): features (shape: BxD, where B is the
                batch size and D the dimensionality of image-question features)

        Returns:
            tf.Tensor: the output scores

        Raises:
            ValueError: if aggregation type or f_phi type specified in config
                is unsupported.
        """

        cfg = self._config['mlp']
        params = cfg['params']
        for l, layer_cfg in enumerate(params):
            layer_name = 'fc{}'.format(l)
            x = fc(x, is_training=self._is_training,
                   name=layer_name, **layer_cfg)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                tf.summary.histogram(
                    layer_name,
                    tf.get_variable('{}/fully_connected/weights'.format(
                        layer_name)),
                    collections=['visualizations']
                )
        return x

    @define_scope
    def _combine_img_w_question(self, img, q):
        """Concatenates image and question embeddings

        Args:
            img (tf.Tensor): the image embedding
            q (tf.Tensor): the question embedding

        Returns:
            tf.Tensor: the concatenated image-question embedding
        """
        img_flat = tf.reshape(
            img, (-1, reduce(mul, img.get_shape().as_list()[1:])))
        return tf.concat((img_flat, q), axis=1)

    @define_scope
    def _output(self, x):
        """Output activation

        Computes the output activation, dummy function for now.

        Args:
            x (tf.Tensor): pre-activation outputs.

        Returns:
            tf.Tensor: output activations
        Raises:
            ValueError: if output type specified in config is unsupported.
        """
        cfg = self._config['output']
        if cfg['type'] == 'categorical':
            pass
        else:
            raise ValueError('unsupported output type specified')
        return x

    @define_scope
    def inference(self, img, q, qlen):
        """Performs the forward propagation (inference)

        Args:
            img (tf.Tensor): a batch of input images (shape BxHxWxC, where
                B is the batch size and H, W, C are height, width and number of
                channels).
            q (tf.Tensor): a batch of input questions (shape BxQ, where
                B is the batch size and Q is the maximum length of questions in
                the batch).
            qlen (tf.Tensor): a batch of input question lengths (shape B, where
                B is the batch size)

        Returns:
            (tf.Tensor, tf.Tensor): output activations and predicted answers (
                argmax of activations)
        """

        # encode image
        img = self._image_encoder(img=img)

        # encode question
        q = self._question_encoder(q=q, qlen=qlen)

        x = self._combine_img_w_question(img, q)

        # classify
        logits = self._mlp(x)
        tf.summary.histogram('logits', logits, collections=['debug'])

        # predict output
        outputs = self._output(logits)
        predicted_answers = tf.argmax(outputs, 1)
        return outputs, predicted_answers

    @define_scope
    def loss(self, img, q, qlen, target_answers):
        """Performs the forward propagation (inference + loss computation)

        Runs inference and computes the loss.

        Args:
            img (tf.Tensor): a batch of input images (shape BxHxWxC, where
                B is the batch size and H, W, C are height, width and number of
                channels).
            q (tf.Tensor): a batch of input questions (shape BxQ, where
                B is the batch size and Q is the maximum length of questions in
                the batch).
            qlen (tf.Tensor): a batch of input question lengths (shape B, where
                B is the batch size)
            target_answers (tf.Tensor): a batch of target answers (shape B,
                where B is the batch size)

        Returns:
            (tf.Tensor, tf.Tensor, tf.Tensor): batch average loss, accuracies
                and predicted answers

        Raises:
            ValueError: if output type specified in config is unsupported.
        """
        predicted_probs, predicted_answers = self.inference(
            img=img, q=q, qlen=qlen)
        cfg = self._config['output']
        if cfg['type'] == 'categorical':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=predicted_probs, labels=target_answers
            )
        else:
            raise ValueError('unsupported output type specified')

        correct_prediction = tf.equal(
            predicted_answers, tf.cast(target_answers, 'int64')
        )
        accuracies = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(loss), accuracies, predicted_answers

# vim: set ts=4 sw=4 sts=4 expandtab:
