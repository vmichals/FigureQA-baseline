#!/usr/bin/env python
#-*- coding: utf-8 -*-

import functools
import tensorflow as tf

def define_scope(function):
    """Automatic variable scope around function"""

    @functools.wraps(function)
    def decorator(self, *args, **kwargs):
        with tf.variable_scope(function.__name__):
            rval = function(self, *args, **kwargs)
        return rval

    return decorator

# vim: set ts=4 sw=4 sts=4 expandtab:
