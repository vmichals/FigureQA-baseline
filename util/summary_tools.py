#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import OrderedDict
from glob import glob
import os

import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError
from tqdm import tqdm


def extract_values(events_file, tags, end_step=None):
    """Extracts values of given tags from an events file

    Args:
        events_file (str): path to an event file
        tags (iterable): the tags for which to extract values
        end_step (int): optional end step (instead of extracting all values
            until the last)

    Returns:
        dict: Dictionary (tags->OrderedDict(step->val))
    """
    summary_iter = tf.train.summary_iterator(events_file)

    values = dict([(tag, OrderedDict()) for tag in tags])

    try:
        for e in summary_iter:
            step = e.step
            if end_step is not None and step > end_step:
                break

            for v in e.summary.value:
                if v.tag in tags:
                    values[v.tag][step] = v.simple_value
    except DataLossError:
        print('data loss at step {}'.format(step))
    return _sort_values_by_step(values)

def extract_values_from_multiple_events_files(events_files, tags, end_step=None):
    """Extracts values from multiple files and merges them

    Args:
        events_files (iterable): paths to events files
        tags (iterable): the tags for which to extract values
        end_step (int): optional end step (instead of extracting all values
            until the last)

    Returns:
        dict: Dictionary (tags->OrderedDict(step->val))
    """
    vals = []
    for events_file in tqdm(events_files):
        vals.append(extract_values(events_file, tags, end_step=end_step))

    vals_merged = vals[0]
    for i in range(1, len(vals)):
        for k, v in vals_merged.items():
            v.update(vals[i][k])

    # sort values by step
    return _sort_values_by_step(vals_merged)


def _sort_values_by_step(vals_dict):
    """Sorts values dictionary by step

    Args:
        vals_dict (dict): a values dictionary as created by extract_values

    Returns:
        dict: values dictionary with sorted steps
    """
    for k, v in vals_dict.items():
        vals_dict[k] = OrderedDict(sorted(v.items()))
    return vals_dict


def extract_values_from_train_dir(train_dir, tags, end_step=None):
    """Extracts values from training directory for given tags

    Args:
        train_dir (str): the training directory to search events files in
        tags (iterable): the tags for which to extract values
        end_step (int): optional end step (instead of extracting all values
            until the last)

    Returns:
        dict: Dictionary (tags->OrderedDict(step->val))
    """
    events_files = glob(os.path.join(train_dir, 'events.out.tfevents.*'))
    vals = extract_values_from_multiple_events_files(events_files,
                                                     tags=tags,
                                                     end_step=end_step)
    return vals


if __name__ == '__main__':
    events_files = [
        'train_dir/RN-FigureQA-20171215-115345/events.out.tfevents.1513367904.cdr26.int.cedar.computecanada.ca',
        'train_dir/RN-FigureQA-20171215-115345/events.out.tfevents.1513632157.cdr36.int.cedar.computecanada.ca',
        'train_dir/RN-FigureQA-20171215-115345/events.out.tfevents.1513367904.cdr26.int.cedar.computecanada.ca',
        'train_dir/RN-FigureQA-20171215-115345/events.out.tfevents.1513632157.cdr36.int.cedar.computecanada.ca',
        'train_dir/RN-FigureQA-20171215-115345/events.out.tfevents.1513637103.cdr211.int.cedar.computecanada.ca',
        'train_dir/RN-FigureQA-20171215-115345/events.out.tfevents.1513839649.cdr206.int.cedar.computecanada.ca',
        'train_dir/RN-FigureQA-20171215-115345/events.out.tfevents.1513877760.cdr39.int.cedar.computecanada.ca',
        'train_dir/RN-FigureQA-20171215-115345/events.out.tfevents.1513880443.cdr39.int.cedar.computecanada.ca',
        'train_dir/RN-FigureQA-20171215-115345/events.out.tfevents.1513962811.cdr26.int.cedar.computecanada.ca',
        'train_dir/RN-FigureQA-20171215-115345/events.out.tfevents.1514231857.cdr35.int.cedar.computecanada.ca',
        'train_dir/RN-FigureQA-20171215-115345/events.out.tfevents.1514669030.cdr162.int.cedar.computecanada.ca'
    ]

    train_dir = 'train_dir/RN-FigureQA-20171215-115345'

    tags = ['train/train_loss', 'train/train_accuracy', 'val/validation_accuracy']

    vals = extract_values_from_train_dir(train_dir=train_dir, tags=tags)






# vim: set ts=4 sw=4 sts=4 expandtab:
