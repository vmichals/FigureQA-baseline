#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import argparse
from glob import glob
import json
import os
import shutil
import tarfile
import time

import matplotlib.pyplot as plt
import numpy as np
from six import iteritems
import tensorflow as tf
from tqdm import tqdm

from data.figureqa import FigureQA
from models.cnn_baseline import CNNBaselineModel
from models.rn import RNModel
from models.textonly_baseline import TextOnlyBaselineModel
from util.summary_tools import extract_values
from util.text_tools import add_line_break_every_k_words
from util.tf_util import parallelize

DEBUG = False
DEBUG_INTERVAL = 100

parser = argparse.ArgumentParser()
parser.add_argument('--train-dir', type=str, required=True,
                    help='experiment folder created by the training script')
parser.add_argument('--data-path', type=str, required=True,
                    help='parent folder of where the data set is located')
parser.add_argument('--tmp-path', type=str,
                    help=('tmp directory where to extract data set (optimally '
                          'on faster storage, if not specified DATA_PATH is '
                          'used)'))
parser.add_argument('--num-gpus', type=int, default=1,
                    help='the number of gpus to use')
parser.add_argument('--log-device-placement',
                    action='store_true', default=False,
                    help=('whether to print the device placement of each '
                          'variable.'))
parser.add_argument('--tfdbg',
                    action='store_true', default=False,
                    help='whether to run the TF debugger')
parser.add_argument('--backup-path', type=str, default='backup_dir',
                    help='backup root dir for long-term storage of parameters')
parser.add_argument('--resume-step', type=int, required=True,
                    help='update step to resume from.')

args = parser.parse_args()

if args.tmp_path is None:
    tmp_path = args.tmp_path
else:
    tmp_path = args.data_path

if args.tfdbg:
    from tensorflow.python import debug as tf_debug

# check experiment folders
assert os.access(args.train_dir, os.W_OK), 'train dir is not writable'
train_dir = args.train_dir
try:
    with open(os.path.join(train_dir, 'val_set')) as fid:
        val_set = fid.read()
except:
    warnings.warn(
        'could not open file {}. using default \'validation2\''.format(
            os.path.join(train_dir, 'val_set')
        ))
    val_set = 'validation2'
backup_train_dir = os.path.join(args.backup_path, os.path.basename(train_dir))
if not os.path.exists(backup_train_dir):
    os.makedirs(backup_train_dir)

assert os.access(backup_train_dir, os.W_OK), 'backup dir is not writable'

# load config file (expects unique config file)
config_files = glob(os.path.join(train_dir, '*_on_figureqa_config.json'))
if len(config_files) == 0:
    raise IOError('no config file found in TRAIN_DIR')
elif len(config_files) > 1:
    raise IOError('more than one config file found in TRAIN_DIR')
config_file = config_files[0]
shutil.copy2(config_file, backup_train_dir)

config = json.load(open(config_file))
model_str = config['model']['name']

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

print('loading training data...')
train_data_object = FigureQA(
    data_path=figureqa_path,
    partition='train1',
    shuffle=True,
    #save_dict_file='figureqa_dict.json',
    load_dict_file='figureqa_dict.json',
    im_size=im_size,
    no_images=no_images
)

dict_size = len(train_data_object.tok_to_idx)

train_data = train_data_object.get_data(
    batch_size=config['batch_size'] * args.num_gpus,
    num_threads=config['num_threads'],
    allow_smaller_final_batch=False
)

print('loading validation data...')
val_data = FigureQA(
    data_path=figureqa_path,
    partition=val_set,
    shuffle=True,
    load_dict_file='figureqa_dict.json',
    im_size=im_size,
    no_images=no_images
).get_data(
    batch_size=config['val_batch_size'] * args.num_gpus,
    num_threads=config['num_threads'],
    allow_smaller_final_batch=False
)

print('restoring model graph...')
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

# create input pipeline
handle = tf.placeholder(tf.string, shape=())
iterator = tf.data.Iterator.from_string_handle(
    handle,
    output_types=train_data.output_types,
    output_shapes=train_data.output_shapes
)
next_batch = iterator.get_next()

# repeat train data indefinitely
train_data = train_data.repeat()
val_data = val_data.repeat()

train_iterator = train_data.make_one_shot_iterator()
val_iterator = val_data.make_one_shot_iterator()

# create the session
sess = tf.Session(
    config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=args.log_device_placement
    )
)
if args.tfdbg:
    sess = tf_debug.LocalCLIDebugWrapperSession(
        sess, dump_root='./dbg')
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

# get handles for train and val set
train_handle = sess.run(train_iterator.string_handle())
val_handle = sess.run(val_iterator.string_handle())

(input_images, input_questions, input_questions_lengths,
 target_answers) = next_batch

# instantiate the model and optimizer
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

opt = tf.train.AdamOptimizer(
    learning_rate=config['learning_rate']
)

# create fprop, bprop, optimization graph
global_step = tf.Variable(args.resume_step, trainable=False)

def compute_grads_fn(**kwargs):
    loss_val, accuracies_vals, predicted_answers = model.loss(
        **{k : v for k, v in iteritems(kwargs)})

    grads = opt.compute_gradients(
        loss_val,
        colocate_gradients_with_ops=True
    )
    return loss_val, accuracies_vals, predicted_answers, grads

def average_grads(grads_list):
    avg_grads = []
    for grads_vars in zip(*grads_list):
        grads = []
        for g, _ in grads_vars:
            grads.append(tf.expand_dims(g, axis=0))
        grad = tf.concat(values=grads, axis=0)
        grad = tf.reduce_mean(grad, axis=0)

        v = grads_vars[0][1]
        grad_and_var = (grad, v)
        avg_grads.append(grad_and_var)
    return avg_grads

input_kwargs = {
    'q': input_questions,
    'qlen': input_questions_lengths,
    'target_answers': target_answers
}
if model_str != 'TextOnlyBaseline':
    input_kwargs['img'] = input_images
loss, accuracies, predicted_answers, grads_list = parallelize(
    fn=compute_grads_fn,
    num_gpus=args.num_gpus,
    **input_kwargs
)

accuracy = tf.reduce_mean(accuracies)
loss = tf.reduce_mean(loss)
grads = average_grads(grads_list)

opt_op = opt.apply_gradients(grads,
                             global_step=global_step)

saver = tf.train.Saver(max_to_keep=100)

# create exponential running average ops for tracking accuracy estimates
ema_train = tf.train.ExponentialMovingAverage(decay=.9, zero_debias=True)
ema_val = tf.train.ExponentialMovingAverage(decay=.9, zero_debias=True)

maintain_train_averages = ema_train.apply((accuracy,))
maintain_val_averages = ema_val.apply((accuracy,))

with tf.control_dependencies((maintain_val_averages,)):
    val_op = ema_val.average(accuracy).read_value()

with tf.control_dependencies((opt_op,)):
    train_op = tf.group(maintain_train_averages)

with tf.control_dependencies((maintain_train_averages,)):
    train_accuracy = ema_train.average(accuracy).read_value()

# create summaries for visualization
tf.summary.scalar('train/train_accuracy', ema_train.average(accuracy),
                  collections=['train', 'accuracies'])
train_summary_op = tf.summary.merge_all('train')

tf.summary.scalar('val/validation_accuracy', ema_val.average(accuracy),
                  collections=['val', 'accuracies'])
val_summary_op = tf.summary.merge_all('val')
vis_summary_op = tf.summary.merge_all('visualizations')

if DEBUG:
    dbg_summary_op = tf.summary.merge_all('debug')

# define some restore ops for vars not covered by saver.restore()

events_file = glob(os.path.join(train_dir, 'events.out*'))[-1]
tags = ['train/train_loss', 'train/train_accuracy', 'val/validation_accuracy']
vals = extract_values(events_file=events_file, tags=tags,
                      end_step=args.resume_step)
best_val_acc = max(list(vals['val/validation_accuracy'].values()))
best_val_step = np.argmax(list(vals['val/validation_accuracy'].values()))

train_avg = sess.graph.get_tensor_by_name(
    'Mean/ExponentialMovingAverage:0')
val_avg = sess.graph.get_tensor_by_name(
    'Mean/ExponentialMovingAverage_1:0')
train_avg_biased = sess.graph.get_tensor_by_name(
    'Mean/ExponentialMovingAverage/biased:0')
val_avg_biased = sess.graph.get_tensor_by_name(
    'Mean/ExponentialMovingAverage_1/biased:0')
train_local_step = sess.graph.get_tensor_by_name(
    'Mean/ExponentialMovingAverage/local_step:0')
val_local_step = sess.graph.get_tensor_by_name(
    'Mean/ExponentialMovingAverage_1/local_step:0')

restore_ops = []
restore_ops.append(tf.assign(
    val_avg,
    list(vals['val/validation_accuracy'].values())[-1]
))
restore_ops.append(tf.assign(
    train_avg,
    list(vals['train/train_accuracy'].values())[-1]
))
# FIXME: quick hack, biased running average initialized with unbiased
restore_ops.append(tf.assign(
    val_avg_biased,
    list(vals['val/validation_accuracy'].values())[-1]
))
restore_ops.append(tf.assign(
    train_avg_biased,
    list(vals['train/train_accuracy'].values())[-1]
))
restore_ops.append((tf.assign(train_local_step, args.resume_step)))
restore_ops.append(tf.assign(val_local_step, args.resume_step))

# some non-tensorboard visualization functions
def visualization(data_handle, save_file, nsamples=5):
    imgs, questions, true_answers, pred_answers = sess.run(
        (input_images, input_questions, target_answers,
            predicted_answers),
        feed_dict={
            handle: data_handle,
            is_training: False
        }
    )
    fig, axes = plt.subplots(nrows=5, ncols=2)
    for i in tqdm(range(10)):
        sample_idx = np.random.randint(len(imgs))
        plt.subplot(5, 2, i + 1)
        plt.imshow(imgs[sample_idx].astype(np.uint8), interpolation='nearest')
        plt.axis('off')
        q = [train_data_object._idx_to_tok[idx]
             for idx in questions[sample_idx]]
        q = ' '.join(q[q.index('<START>') + 1:q.index('<END>')])

        pred_a = train_data_object._unique_answers[pred_answers[sample_idx]]
        gt_a = train_data_object._unique_answers[true_answers[sample_idx]]
        q = add_line_break_every_k_words(q, 7)
        plt.title('{q}? \npred. answer: {pred_a} \ntrue answer: {gt_a}'.format(
            q=q, pred_a=pred_a, gt_a=gt_a), fontdict={'fontsize': 6})
    fig.tight_layout()
    fig.savefig(save_file, dpi=300)

def draw_profiling_plots(**kwargs):
    nargs = len(kwargs)
    plt.clf()
    for i, (argname, argval) in enumerate(iteritems(kwargs)):
        ax = plt.subplot(1, nargs, i+1)
        ax.plot(range(len(argval)), argval)
        ax.xlabel('step')
        ax.ylabel(argval)
    plt.savefig(os.path.join(train_dir, 'profiling_plot.pdf'))

# restore vars
init_op = tf.global_variables_initializer()
sess.run(init_op)
saver.restore(sess, os.path.join(train_dir, 'model-{}'.format(args.resume_step)))
sess.run(restore_ops)
summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

print('resuming training...')
sec_over_batch_list = []
samples_per_sec_list = []
for step in range(args.resume_step, config['max_num_steps']):
    if DEBUG and (step + 1) % DEBUG_INTERVAL == 0:
        dbg_summary = sess.run((dbg_summary_op), feed_dict={
            handle: val_handle,
            is_training: False
        })
        summary_writer.add_summary(dbg_summary, global_step=step)
    start = time.time()
    # train and update train accuracy running avg
    loss_val, _, train_summary, train_acc = sess.run(
        (loss, train_op, train_summary_op, train_accuracy),
        feed_dict={handle: train_handle, is_training: True}
    )
    step_time = max(time.time() - start, .001)

    # update val accuracy running avg
    val_acc, val_summary = sess.run((val_op, val_summary_op), feed_dict={
        handle: val_handle,
        is_training: False
    })
    summary_writer.add_summary(train_summary, global_step=step)
    summary_writer.add_summary(val_summary, global_step=step)

    if step >= 99 and (step + 1) % config['val_interval'] == 0:
        if val_acc > best_val_acc:
            # update best val accuracy
            best_val_step = step
            best_val_acc = val_acc
            print('found new best validation accuracy {} at step {}'.format(
                best_val_acc, best_val_step + 1))
            saver.save(sess, os.path.join(train_dir, 'model_val_best'))
            print('backing up val best...')
            fnames = glob(os.path.join(backup_train_dir, 'model_val_best*'))
            # do not backup backup files
            fnames = list(filter(lambda s: not s.endswith('_bak'), fnames))
            for fname in fnames:
                shutil.copy2(
                    fname,
                    os.path.join(backup_train_dir, '{}_bak'.format(
                        os.path.basename(fname)))
                )
            json.dump(step, open(
                os.path.join(backup_train_dir, 'model_val_best_step.json'), 'w'))
            for fname in glob(os.path.join(train_dir, 'model_val_best.*')):
                shutil.copy(fname, backup_train_dir)
            print('done')

    samples_per_sec = (args.num_gpus * config['batch_size']) / step_time

    sec_over_batch_list.append(step_time)
    samples_per_sec_list.append(samples_per_sec)
    print(
        (
            'step {step:d}, train loss {train_loss:.2f}, '
            'train acc {train_acc:.2f}, val acc {val_acc:.2f} '
            '(best val acc so far {best_val_acc:.2f} '
            '(at step {best_val_step:d}) - '
            '({step_time:.3f} sec/batch, '
            '{samples_per_sec:.3f} samples/sec)'
        ).format(
            step=step + 1, train_loss=loss_val,
            train_acc=train_acc, val_acc=val_acc,
            best_val_acc=best_val_acc, best_val_step=best_val_step + 1,
            step_time=step_time,
            samples_per_sec=samples_per_sec
        ))
    if (step + 1) % config['save_interval'] == 0:
        saver.save(sess, os.path.join(train_dir, 'model'), global_step=step + 1)
        # also backup events files
        for fname in glob(os.path.join(train_dir, 'events.out*')):
            shutil.copy2(fname, backup_train_dir)

    if (step + 1) % config['visualization_interval'] == 0:
        if not no_images:
            vis_summary = sess.run(vis_summary_op)
            summary_writer.add_summary(vis_summary, global_step=step)
            # generate some VQA samples
            visualization(train_handle, os.path.join(
                train_dir, 'train_visualizations.pdf')
            )
            visualization(val_handle, os.path.join(
                train_dir, 'val_visualizations.pdf')
            )
        #draw_profiling_plots(
        #    samples_per_sec=samples_per_sec_list,
        #    sec_over_batch=sec_over_batch_list
        #)

print('done')

sess.close()

# vim: set ts=4 sw=4 sts=4 expandtab:
