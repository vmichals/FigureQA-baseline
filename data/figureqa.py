#!/usr/bin/env python
#-*- coding: utf-8 -*-

import json
import os
import warnings

import numpy as np
from six import iteritems
import tensorflow as tf
from tqdm import tqdm

from util import array_tools, text_tools


class FigureQA(object):
    def __init__(self, data_path, partition, questions_file=None,
                 annotations_file=None, shuffle=False, im_size=None,
                 load_dict_file=None, save_dict_file=None, no_images=False):
        """

        Args:
            data_path (str): directory containing the extracted FigureQA data set
            partition (str): one of ('train1', 'validation1', 'validation2',
                'test1', 'test2')
            questions_file (str): path to qa_pairs.json, if not in default
                location
            annotations_file (str): path to annotations.json, if not in default
                location
            shuffle (bool): whether to shuffle samples
            im_size (tuple): containing target height and width, images will be
                resized, such that the longer side matches the target length,
                after which the image is padded to the target size
            load_dict_file (str): path to a dictionary file, if one was dumped in
                a prior run
            save_dict_file (str): file path for dumping the generated dictionary
            no_images (bool): whether to omit images

        Raises:
            IOError: if dictionary could not be loaded
        """
        tokenizer_kwargs = {
            'keep_punctuation':  (',', ';'),
            'drop_punctuation': ('.', '?')
        }
        assert load_dict_file is not None or save_dict_file is not None

        if partition == 'test1': partition = 'no_annot_test1'
        elif partition == 'test2': partition = 'no_annot_test2'
        self._partition = partition
        if questions_file is None:
            questions_file = os.path.join(data_path, partition, 'qa_pairs.json')
        self._questions_file = questions_file
        if annotations_file is None:
            annotations_file = os.path.join(data_path, partition,
                                            'annotations.json')
        self._annotations_file = annotations_file
        self._shuffle = shuffle
        self._im_size = im_size
        self._no_images = no_images

        with open(self._questions_file) as qfile:
            annotation = json.load(qfile)
        self._images_dir = os.path.join(
            data_path, partition, 'png')

        if 'test' in self._partition:
            self._image_indices, self._questions, self._question_ids = zip(
                *((q['image_index'], q['question_string'], q['question_id'])
                  for q in annotation['qa_pairs'])
            )
        else:
            (self._image_indices, self._questions, self._question_ids,
             self._answers) = zip(
                 *((q['image_index'], q['question_string'], q['question_id'],
                    q['answer'])
                   for q in annotation['qa_pairs'])
             )

        self._image_fnames = ['{}.png'.format(idx)
                              for idx in self._image_indices]
        self._image_fnames = np.array(self._image_fnames)
        self._questions = np.array(self._questions)
        self._question_ids = np.array(self._question_ids)
        self._image_indices = np.array(self._image_indices)
        if hasattr(self, '_answers'):
            self._answers = np.array(self._answers)

        # do initial shuffle to avoid TF only shuffling inside batches of
        # mostly similar samples
        if self._shuffle:
            perm = np.random.permutation(len(self._image_fnames))
            self._image_fnames = self._image_fnames[perm]
            self._image_indices = self._image_indices[perm]
            self._questions = self._questions[perm]
            self._question_ids = self._question_ids[perm]
            self._question_indices = perm
            if hasattr(self, '_answers'):
                self._answers = self._answers[perm]
        else:
            self._question_indices = np.arange(len(self._image_fnames))

        if load_dict_file is None:
            print('building dictionary...')
            self._tok_to_idx = text_tools.build_dict(
                documents=self._questions,
                min_occurences=1,
                **tokenizer_kwargs
            )
            print('dumping dictionary to {}...'.format(save_dict_file))
            json.dump(self._tok_to_idx, open(save_dict_file, 'w'))
        else:
            try:
                print('trying to load dictionary from {}...'.format(
                    load_dict_file))
                self._tok_to_idx = json.load(open(load_dict_file))
            except IOError:
                raise IOError('failed to load dictionary from {}'.format(
                    load_dict_file))

        print('building inverse dictionary...')
        self._idx_to_tok = text_tools.invert_dict(self._tok_to_idx)
        print('tokenizing questions...')
        questions_tokenized = [
            text_tools.tokenize(q, **tokenizer_kwargs)
            for q in tqdm(self._questions)]
        self._questions_ints = [
            [self._tok_to_idx[tok] for tok in q]
            for q in tqdm(questions_tokenized)
        ]
        self._question_lens = np.array([len(q) for q in self._questions_ints])
        self._max_q_len = max(self._question_lens)

        if hasattr(self, '_answers'):
            self._unique_answers = ['no', 'yes']
            self.answer_to_idx = lambda a: self._unique_answers.index(a)
            self.idx_to_answer = lambda idx: self._unique_answers[idx]

    def get_data(self, batch_size, num_threads=1, buffer_size=None,
                 allow_smaller_final_batch=True,
                 return_indices=False, return_question_strings=False):
        """Returns a batched tf.data.Dataset

        Args:
            batch_size (int): the number of samples in a minibatch
            num_threads (int): the number of threads used for preprocessing
            buffer_size (int or None): the number of samples to sample from,
                when using shuffling (defaults to 20*batch_size if set to None)
            allow_smaller_final_batch (bool): whether to allow the last batch
                to be smaller than batch_size. Parallel computation has to
                take non-fixed batch_sizes into consideration, if set to True.
                In practice it is easier to just shuffle training data and skip
                the last batch (i.e. set allow_smaller_final_batch=False)
            return_indices (bool): whether to also return question_indices,
                image_indices and question_ids (needed for test evaluation, see
                http://datasets.maluuba.com/FigureQA/evaluation )
            return_question_strings (bool): whether to also return the question
                strings
        """

        # we don't want to pad all questions in the data set, we do that later
        # on-the-fly for each batch. While memory efficient, this requires
        # using sparse tensors
        print('creating sparse tensor for questions...')
        indices, values, dense_shape = \
            array_tools.nested_2d_list_to_sparse_array(
                self._questions_ints, ndim=self._max_q_len
            )
        questions_sparse_tensor = tf.SparseTensor(
            indices=indices, dense_shape=dense_shape,
            values=values
        )

        print('adding image filenames and questions to sub dataset list...')
        sub_datasets = [
            tf.data.Dataset.from_tensor_slices(self._image_fnames),
            tf.data.Dataset.from_sparse_tensor_slices(
                questions_sparse_tensor
            ),
            tf.data.Dataset.from_tensor_slices(self._question_lens)
        ]

        if hasattr(self, '_answers'):
            print('adding answers to sub dataset list...')
            sub_datasets.append(
                tf.data.Dataset.from_tensor_slices(self._answers)
            )
        if return_indices:
            print('adding question_indices, image_indices and question_ids to '
                  'sub dataset list...')
            sub_datasets.extend([
                tf.data.Dataset.from_tensor_slices(self._question_indices),
                tf.data.Dataset.from_tensor_slices(self._image_indices),
                tf.data.Dataset.from_tensor_slices(self._question_ids)
            ])
        if return_question_strings:
            sub_datasets.append(
                tf.data.Dataset.from_tensor_slices(self._questions)
            )

        print('zipping up sub datasets...')
        # zip together all the sub data sets
        dataset = tf.data.Dataset.zip(
            tuple(sub_datasets)
        )

        def input_parser(im_fname, *args):
            assert len(args) in (2, 3, 5, 6, 7), 'unexpected number of args'
            q_indices, q_vals, q_out_shp = args[0]
            args = tuple([
                tf.sparse_to_dense(sparse_indices=q_indices,
                                   sparse_values=q_vals,
                                   output_shape=q_out_shp)] + list(args[1:]))

            # if no images should be returned, we just return their filenames
            if self._no_images:
                im = im_fname
            else:
                im_file = tf.read_file(self._images_dir + '/' + im_fname)
                im = tf.image.decode_png(im_file, channels=3)
                if self._im_size is not None:
                    h, w = tf.unstack(tf.cast(tf.shape(im)[:2], tf.float32))
                    target_h, target_w = self._im_size
                    aspect_ratio = w / h
                    w_ratio = target_w / w
                    h_ratio = target_h / h
                    target_aspect_ratio = float(target_w) / target_h
                    # ar > tar:
                    # scale w, h shorter
                    # ar < tar:
                    # scale h, w shorter
                    im = tf.cond(
                        tf.greater(aspect_ratio, target_aspect_ratio),
                        lambda: tf.image.resize_images(
                            im, size=(tf.cast(w_ratio * h, tf.int32),
                                    tf.cast(target_w, tf.int32))
                        ),
                        lambda: tf.image.resize_images(
                            im, size=(tf.cast(target_h, tf.int32),
                                    tf.cast(h_ratio * w, tf.int32))
                        )
                    )
                    im = tf.image.resize_image_with_crop_or_pad(
                        im, target_height=target_h, target_width=target_w
                    )
                    im.set_shape((target_h, target_w, 3))
                    im = im / 255.

            return (im, ) + args

        if self._shuffle:
            print('shuffling dataset...')
            if buffer_size is None:
                buffer_size = 20 * batch_size
            dataset = dataset.shuffle(buffer_size=buffer_size)
        print('adding input parser to dataset pipeline')
        dataset = dataset.map(
            input_parser,
            num_parallel_calls=num_threads
        ).prefetch(num_threads * batch_size)
        print('batching...')
        dataset = dataset.batch(batch_size)
        if not allow_smaller_final_batch:
            dataset = dataset.filter(
                lambda imgs, *args: tf.equal(
                    tf.shape(imgs)[0],
                    batch_size
                )
            )
        return dataset

    @property
    def tok_to_idx(self):
        """The word to index dictionary"""
        return self._tok_to_idx

    @property
    def idx_to_tok(self):
        """The index to word dictionary"""
        return self._idx_to_tok

# vim: set ts=4 sw=4 sts=4 expandtab:
