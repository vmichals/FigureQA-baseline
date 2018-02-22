#!/usr/bin/env python
#-*- coding: utf-8 -*-

from six import iteritems

import tensorflow as tf

def parallelize(fn, num_gpus, **kwargs):
    """Parallelizes a tensorflow function

    Args:
        fn (callable): A function taking keywords arguments
        num_gpus (int): The number of GPUs to parallelize on.
        kwargs: Keywords arguments for fn. The values should be tensors, because
            they have to be split into num_gpus parts.

    Returns:
        list: A list of tensors containing the concatenated results of the
            parallel function calls.

    """
    parts = {}
    for k, v in iteritems(kwargs):
        parts[k] = tf.split(v, num_gpus)

    output = []
    for g in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type='GPU', device_index=g)):
            # all gpus use vars of gpu 0
            with tf.variable_scope(tf.get_variable_scope(), reuse=g>0):
                output.append(
                    fn(**{k : v[g] for k, v in iteritems(parts)})
                )
    output = [outp for outp in zip(*output)]
    concat_output = []
    for outp in output:
        # can't concat scalars, so use stack instead
        if isinstance(outp[0], list):
            concat_output.append(outp)
        elif outp[0].get_shape().ndims == 0:
            concat_output.append(tf.stack(outp))
        else:
            concat_output.append(tf.concat(outp, axis=0))
    return concat_output


# vim: set ts=4 sw=4 sts=4 expandtab:
