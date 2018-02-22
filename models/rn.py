#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import division

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops import embedding_ops, rnn
from tensorflow.python.framework.ops import IndexedSlices

from .ops import conv2d, fc
from util.array_tools import repeat
from util.decorators import define_scope
from util.plot_tools import get_conv1_filter_grid_img


class RNModel(object):
    """Relation Network

    Implements the Relation Network, introduced by Santoro, Adam, et al. "A
    simple neural network module for relational reasoning." Advances in neural
    information processing systems. 2017.

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
    def _select_blocks(self, x):
        """Select blocks from which to choose pairs

        Args:
            x (tf.Tensor): the input from which to select objects (called blocks
                here)

        Returns:
            tf.Tensor: the selected blocks (shape BxNxD, where B is the batch
                size, N the number of selected blocks and D the dimensionality
                of the block descriptors)

        Raises:
            ValueError: if the block selection type specified in the config is
                unsupported.
        """
        cfg = self._config['block_selection']
        if cfg['type'] == 'dense':
            assert x.shape.ndims == 4, \
                'for now only 4d tensors supported in block selection'
            batch_size = tf.shape(x)[0]
            _, h, w, num_channels = x.get_shape().as_list()
            blocks = []
            indices = tf.range(h * w)
            row_indices = indices // w
            col_indices = indices % w
            # create tensor containing the block coordinates
            block_coords = tf.cast(tf.tile(
                tf.concat(
                    (row_indices[:, None] / h, col_indices[:, None] / w),
                    axis=1
                )[None],
                multiples=(batch_size, 1, 1)
            ), tf.float32)
            # merge spatial dimensions
            blocks = tf.concat(
                (
                    tf.reshape(x, shape=(batch_size, h * w, num_channels)),
                    block_coords
                ), axis=2
            )
        else:
            raise ValueError('unsupported block selection type specified')
        return blocks

    @define_scope
    def _pair_blocks(self, blocks):
        """Pairs blocks to be fed to g_theta

        Args:
            blocks (tf.Tensor): contains the blocks lists (shape BxNxD as
                returned by _select_blocks())

        Returns:
            tuple of tf.Tensor: tuple of blocksA, blocksB, where
                blocksA[:, p, :] (blocksB[:, p, :]) contain the first (second)
                elements of the p-th pair (shape B, P, K, where B is the batch
                size, P the number of pairs, and K the dimensionality of
                descriptor pairs)
        """
        n = blocks.get_shape().as_list()[1]

        blocksA = tf.tile(blocks, multiples=(1, n, 1))

        blocksB = repeat(
            blocks, repeats=n, axis=1)
        return (blocksA, blocksB)

    @define_scope
    def _f_phi(self, x):
        """Computes the prediction by aggregating relational features

        Aggregates relational features computed by g_theta on pairs and computes
        a prediction.

        Args:
            x (tf.Tensor): relational features (shape: BxPxR, where B is the
                batch size, P the number of pairs and R the dimensionality of
                relational features)

        Returns:
            tf.Tensor: the RN prediction

        Raises:
            ValueError: if aggregation type or f_phi type specified in config
                is unsupported.
        """
        cfg = self._config['f_phi']
        params = cfg['params']
        if cfg['type'] == 'mlp':
            if cfg['aggregation_type'] == 'avg':
                x = tf.reduce_mean(x, axis=1)
            else:
                raise ValueError('unsupported aggregation type specified')
            for l, layer_cfg in enumerate(params):
                layer_name = 'fc{}'.format(l)
                x = fc(x, is_training=self._is_training,
                       weights_initializer=tf.truncated_normal_initializer(
                           stddev=0.1),
                       biases_initializer=tf.constant_initializer(.1),
                       name=layer_name, **layer_cfg)
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    tf.summary.histogram(
                        layer_name,
                        tf.get_variable('{}/fully_connected/weights'.format(
                            layer_name)),
                        collections=['debug']
                    )
        else:
            raise ValueError('unsupported f_phi type specified')
        return x

    @define_scope
    def _g_theta(self, blocks, cond_info):
        """Computes relational features for block pairs

        Args:
            blocks (tf.Tensor): the object pairs as returned by _pair_blocks()
            cond_info (tf.Tensor): the conditioning info, appended to pair
                descriptors before processing (e.g. a question embedding for QA
                tasks).

        Returns:
            tf.Tensor: relational features (shape BxPxR, where B is the batch
                size, P the number of pairs and R the dimensionality of the last
                layer of g_theta)

        Raises:
            ValueError: if the g_theta type specified in the config is
                unsupported.
        """
        cfg = self._config['g_theta']
        params = cfg['params']
        if cfg['type'] == 'mlp':

            # get number of block combinations
            batch_size = tf.shape(blocks[0])[0]
            n = blocks[0].get_shape().as_list()[1]

            cond_info = tf.tile(cond_info[:, None], (1, n, 1))
            x = tf.concat(
                blocks + (cond_info,),
                axis=2
            )

            #x = tf.reshape(x, (batch_size * n, -1))
            x = tf.reshape(x, (batch_size * n, x.get_shape().as_list()[2]))
            for l, layer_cfg in enumerate(params):
                layer_name = 'fc{}'.format(l)
                x = fc(x, is_training=self._is_training,
                       name=layer_name,
                       weights_initializer=tf.truncated_normal_initializer(
                           stddev=0.1),
                       biases_initializer=tf.constant_initializer(.1),
                       **layer_cfg)
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    tf.summary.histogram(
                        layer_name,
                        tf.get_variable('{}/fully_connected/weights'.format(
                            layer_name)),
                        collections=['debug']
                    )
            # reshape back to separate batch and nblocks axes
            x = tf.reshape(x, (batch_size, n, layer_cfg['num_outputs']))
        else:
            raise ValueError('unsupported g_theta type specified')
        return x

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

        # select blocks & attach coordinates
        blocks = self._select_blocks(img)
        tf.summary.histogram('blocks', blocks, collections=['debug'])

        # pair blocks and combine with conditional info (e.g. a question)
        blocks = self._pair_blocks(blocks)
        tf.summary.histogram('block_pairs', blocks, collections=['debug'])

        # run relational module
        rn_features = self._g_theta(blocks, cond_info=q)
        tf.summary.histogram('rn_features', rn_features, collections=['debug'])

        # aggregate relational features
        logits = self._f_phi(rn_features)
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
