#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from glob import glob
import json
import os
import re
import shutil
import tarfile

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from data.figureqa import FigureQA
from models.cnn_baseline import CNNBaselineModel
from models.rn import RNModel
from models.textonly_baseline import TextOnlyBaselineModel

DEBUG = False
TFDBG = False
if DEBUG and TFDBG:
    from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, required=True,
                    help='parent folder of where the data set is located')
parser.add_argument('--tmp-path', type=str,
                    help=('tmp directory where to extract data set (optimally '
                          'on faster storage, if not specified DATA_PATH is '
                          'used)'))
parser.add_argument('--train-dir', type=str, required=True,
                    help='training directory of the trained model.')
parser.add_argument('--meta-file', type=str,
                    help='path to the meta-file from which to load parameters.')
parser.add_argument('--partition', type=str, default='test2',
                    help='name of the partition to evaluate the model on.')
parser.add_argument('--batchsize', type=int, default=64,
                    help='Size of the mini-batches.')

args = parser.parse_args()

train_dir = args.train_dir

# load config file (expects unique config file)
config_files = glob(os.path.join(train_dir, '*_on_figureqa_config.json'))
if len(config_files) == 0:
    raise IOError('no config file found in TRAIN_DIR')
elif len(config_files) > 1:
    raise IOError('more than one config file found in TRAIN_DIR')
config_file = config_files[0]

config = json.load(open(config_file))
model_str = config['model']['name']

if args.meta_file is None:
    meta_file = os.path.join(train_dir, 'model_val_best.meta')
    assert os.access(meta_file, os.R_OK)
else:
    meta_file = args.meta_file

# copy FigureQA to tmp path, if not done before
figureqa_src_path = os.path.join(args.data_path, 'FigureQA')
figureqa_path = os.path.join(args.tmp_path, 'FigureQA')
assert not os.system('mkdir -p {}'.format(figureqa_path))
copy_done_file = os.path.join(figureqa_path, '.done')
same_path = os.path.samefile(figureqa_src_path, figureqa_path)
copy_done = os.path.isfile(copy_done_file)
if not (copy_done or same_path):
    for fname in glob(os.path.join(figureqa_src_path, '*.tar.gz')):
        shutil.copy2(fname, figureqa_path)
if not copy_done:
    for fname in glob(os.path.join(figureqa_path, '*.tar.gz')):
        print('extracting {}...'.format(fname))
        f = tarfile.open(fname)
        f.extractall(path=figureqa_path)
        # if figureqa_path not equal to figureqa_src_path, we delete the
        # copy of the archive
        if not same_path:
            os.remove(fname)
    assert not os.system('touch {}'.format(copy_done_file))


no_images = model_str == 'TextOnlyBaseline'

if no_images:
    im_size = None
else:
    im_size = config['im_size']

print('loading {} data...'.format(args.partition))
data_object = FigureQA(
    data_path=figureqa_path,
    partition=args.partition,
    shuffle=False,
    load_dict_file='figureqa_dict.json',
    im_size=im_size,
    no_images=no_images
)
nsamples = len(data_object._questions)
dict_size = len(data_object.tok_to_idx)

if 'test' in args.partition:
    # check if answers available. If not, just dump predictions...
    try:
        answers_df = pd.read_csv(
            os.path.join(figureqa_path, args.partition, 'answers.csv'), index_col=0)
        answers = answers_df['answer'].as_matrix()
    except:
        answers = None
else:
    # not test partition, so true answers are collected as we iterate through
    # the data
    answers = []

data = data_object.get_data(batch_size=args.batchsize, return_indices=True,
                            return_question_strings=True)

# create the session
sess = tf.Session()
if DEBUG and TFDBG:
    sess = tf_debug.LocalCLIDebugWrapperSession(
        sess, dump_root='/data/home/vmichals/data/tfdbg')
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

print('restoring model graph...')
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

iterator = data.make_one_shot_iterator()
next_batch = iterator.get_next()
if answers == []:
    (input_images, input_questions, input_questions_lengths,
     answers_batch, question_indices, image_indices, question_ids,
     question_strings) = next_batch
    iter_answers = True
else:
    (input_images, input_questions, input_questions_lengths, question_indices,
     image_indices, question_ids, question_strings) = next_batch
    iter_answers = False

# instantiate the model
if model_str == 'RN':
    model = RNModel(
        is_training=is_training,
        config=config['model'],
        dictionary_size=dict_size
    )
elif model_str == 'CNNBaseline':
    model = CNNBaselineModel(
        is_training=is_training,
        config=config['model'],
        dictionary_size=dict_size
    )
elif model_str == 'TextOnlyBaseline':
    model = TextOnlyBaselineModel(
        is_training=is_training,
        config=config['model'],
        dictionary_size=dict_size
    )
else:
    raise ValueError('unsupported model type found in config file')

input_kwargs = {
    'q': input_questions,
    'qlen': input_questions_lengths,
}
if model_str != 'TextOnlyBaseline':
    input_kwargs['img'] = input_images
# NOTE: workaround, during training inference is nested inside
#       the loss variable scope
with tf.variable_scope('loss'):
    _, predictions = model.inference(**input_kwargs)

print('create lists of indices for dumping predictions.')
prediction_list = []
q_index_list = []
im_index_list = []
q_id_list = []
q_str_list = []
idx = 0
batch_cnt = 0
nbatches = int(np.ceil(float(nsamples) / args.batchsize))
print('nbatches: {0}'.format(nbatches, ))
print('nsamples: {0}'.format(nsamples, ))

dir(tf.contrib)
saver = tf.train.Saver(max_to_keep=100)
saver.restore(sess, os.path.splitext(meta_file)[0])

if answers is not None:
    accuracy = 0.
    preds_correct_list = []

# loop over all batches until tf.errors.OutOfRangeError
while True:
    try:
        print('fetching batch {}/{}'.format(batch_cnt + 1, nbatches))

        # fetch predictions
        if iter_answers:
            (preds, answers_batch_np, q_indices, im_indices, q_ids,
             q_strs)  = sess.run(
                (predictions, answers_batch, question_indices, image_indices,
                 question_ids, question_strings), feed_dict={
                    is_training: False,
                }
            )

            answers.extend(answers_batch_np)
        else:
            (preds, q_indices, im_indices, q_ids, q_strs) = sess.run(
                (predictions, question_indices, image_indices,
                 question_ids, question_strings), feed_dict={
                    is_training: False,
                }
            )
        nbatch_samples = len(preds)
        prediction_list.extend(preds.astype(bool).tolist())
        q_index_list.extend(q_indices)
        im_index_list.extend(im_indices)
        q_id_list.extend(q_ids)
        q_str_list.extend(q_strs)

        # compare with answers
        if answers is not None:
            preds_correct = answers[idx:idx+nbatch_samples] == preds.astype(bool)
            preds_correct_list.extend(preds_correct)
            accuracy = (accuracy * idx + np.sum(preds_correct)) / (
                idx + nbatch_samples)
        idx += nbatch_samples
        batch_cnt += 1
    except tf.errors.OutOfRangeError:
        break

# if there are ground truth answers, print and save accuracy
if answers is not None:
    print('\naccuracy on "{1}": {0}'.format(accuracy, args.partition))
    with open('accuracy_{0}_{1}_{2}_{3}.txt'.format(
        args.partition,
        config['model']['name'],
        config['dataset']['name'],
        os.path.splitext(os.path.split(meta_file)[1])[0]), 'w') as fid:
        json.dump(accuracy, fid)

# dump predictions
lines = ['question_index,image_index,question_id,question_string,answer\n']

for i in range(len(prediction_list)):
    lines.append('{},{},{},{},{}\n'.format(q_index_list[i],
                                           im_index_list[i],
                                           q_id_list[i],
                                           q_str_list[i].decode('UTF-8'),
                                           int(prediction_list[i])))

predictions_fname = 'preds_{0}_{1}_{2}_{3}.csv'.format(
    args.partition,
    config['model']['name'],
    config['dataset']['name'],
    os.path.splitext(os.path.split(meta_file)[1])[0]
)
with open(predictions_fname, 'w') as fid:
    fid.writelines(lines)

if answers is None:
    print(
        ('Dumped {} predictions to {}, to get the accuracy, please send the file'
         'to figureqa@microsoft.com with a short comment.').format(
             args.partition, predictions_fname)
    )

# vim: set ts=4 sw=4 sts=4 expandtab:
