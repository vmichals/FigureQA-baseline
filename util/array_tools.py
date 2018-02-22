#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import tensorflow as tf


def traverse(item):
    """Generator that returns the elements of a nested iterable

    Args:
        item (iterable or any type): If item is iterable, traverse it
            recursively, else yield item

    Yields:
        The next element (in postorder).
    """
    try:
        for i in iter(item):
            for j in traverse(i):
                yield j
    except TypeError:
        yield item

def repeat(x, repeats, axis=0):
    """Repeat a tensor N times along an axis.

    Args:
        x (tf.Tensor): the tensor to repeat.
        repeats (int): the number of times to repeat x.
        axis (int): The axis along which to repeat x.

    Returns:
        tf.tensor:
    """
    ndim = x.shape.ndims

    n = x.get_shape().as_list()[axis]
    if n is None:
        idx = tf.range(tf.shape(x)[axis])
    else:
        idx = tf.range(n)
    idx = tf.reshape(idx, (-1, 1))
    idx = tf.tile(idx, (1, repeats))
    idx = tf.reshape(idx, (-1,))
    return tf.gather(x, idx, axis=axis)

# TODO: generalize to n dimensions
def nested_2d_list_to_sparse_array(lst, ndim):
    """Converts nested 2d list to indices, values and shape of a sparse tensor

    Args:
        lst (list): The nested list to be converted.
        ndim (int): The maximum length of a row.

    Returns:
        tuple of (list, list, tuple): The index list non-zero elements,
            a list of the corresponding values and the shape of the dense
            matrix.
    """
    # get lengths of rows
    lens = map(len, lst)
    # get flat list of values
    values = [x for x in traverse(lst)]
    shape = (len(lst), ndim)

    # create index list
    indices = [x for x in itertools.chain.from_iterable(
        [zip((row,) * l, range(l)) for row, l in enumerate(lens)])]
    return indices, values, shape


if __name__ == '__main__':
    import numpy as np

    lst = [[1,2,3,4], [5,6],[10]]
    ndim = 5

    indices, values, shape = nested_2d_list_to_sparse_array(
        lst, ndim=ndim)

    x = np.zeros((len(lst), ndim), dtype=int)
    for i, val in zip(indices, values):
        x[i] = val

    print(x)


# vim: set ts=4 sw=4 sts=4 expandtab:
