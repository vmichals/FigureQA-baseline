#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import Counter, OrderedDict
from math import ceil

from six import iteritems
from tqdm import tqdm


# add special tokens here, with integer values 0, 1, ...
SPECIAL_TOKENS = OrderedDict(
    (
        ('<NULL>', 0),
        ('<START>', 1),
        ('<END>', 2),
        ('<UNK>', 3),
    )
)


def tokenize(s, keep_punctuation=(), drop_punctuation=(), delimiter=' ',
             add_start=True, add_end=True):
    """Tokenizes a sentence

    Args:
        s (str): the sentence
        keep_punctuation (tuple): contains the punctuation to keep
        drop_punctuation (tuple): contains the punctuation to drop
        delimiter (str): the delimiter between tokens
        add_start (bool): whether to add a <START> token
        add_end (bool): whether to add a <END> token

    Returns:
        list: A list of tokens.
    """
    for kp in keep_punctuation:
        s = s.replace(kp, '{0}{1}{0}'.format(delimiter, kp))
    for rp in drop_punctuation:
        s = s.replace(rp, '')
    tokens = s.lower().split(delimiter)
    tokens = [t.strip() for t in tokens if len(t.split()) > 0]
    if add_start:
        tokens.insert(0, '<START>')
    if add_start:
        tokens.append('<END>')

    return tokens


def build_dict(documents, min_occurences, delimiter=' ',
               keep_punctuation=(),
               drop_punctuation=()):
    """Builds a word dictionary from a list of documents

    Args:
        documents (list): list of strings containing sentences
        min_occurences (int): minimum occurences for a dictionary entry
        keep_punctuation (tuple): contains the punctuation to keep
        drop_punctuation (tuple): contains the punctuation to drop

    Returns:
        OrderedDict: The generated token to index dictionary.
    """
    tokens = []
    for doc in tqdm(documents):
        tokens.extend(tokenize(doc, add_start=False, add_end=False,
                               keep_punctuation=keep_punctuation,
                               drop_punctuation=drop_punctuation))

    for tok in SPECIAL_TOKENS:
        try:
            tokens.remove(tok)
        except ValueError:
            pass

    counter = Counter(tokens)

    token_to_idx = SPECIAL_TOKENS.copy()
    for token, cnt in iteritems(counter):
        if cnt >= min_occurences:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def invert_dict(d):
    """Inverts a dictionary (values are swapped with keys)

    Args:
        d (dictionary): the dictionary to invert

    Returns:
        OrderedDict: The input dictionary with swapped values and keys.
    """
    return OrderedDict(((v, k) for k, v in iteritems(d)))


def add_line_break_every_k_words(s, k):
    """Inserts a line break every k words in a string

    Args:
        s (str): the string
        k (int): the number of words in each line

    Returns:
        str: The input string with line breaks inserted after every k-th word.
    """
    s = s.split()
    s = [' '.join(s[i * k: (i + 1) * k])
         for i in range(int(ceil(len(s) / float(k))))]
    return '\n'.join(s)


if __name__ == '__main__':
    from pprint import pprint

    print(add_line_break_every_k_words(
        'a b c d e f g h i j k l m n o p q r s t u v w x y z', 5))

    pprint(tokenize(
        'test sentence test abc hallo abc test. abc;asd,dsadadq. asda?dasa  ',
        keep_punctuation=(',', ';'), drop_punctuation=('.', '?')))

    docs = [
        'test sentence test abc hallo abc test. abc;asd,dsadadq. asda?dasa  ',
        'sfdafh wfafl sentence, test, hallo ; ? abc asd'
    ]

    d = build_dict(documents=docs, min_occurences=1)
    inv_d = invert_dict(d)

    pprint(dict(d.items()))
    pprint(dict(inv_d.items()))

# vim: set ts=4 sw=4 sts=4 expandtab:
