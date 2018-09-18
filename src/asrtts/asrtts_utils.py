#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import logging
import os

# chainer related
import chainer
from chainer import training
from chainer.training import extension

from asr_utils import make_batchset

# io related
import kaldi_io_py

# matplotlib related
import matplotlib
matplotlib.use('Agg')


def pad_ndarray_list(batch, pad_value):
    """FUNCTION TO PERFORM PADDING OF NDARRAY LIST
    :param list batch: list of the ndarray [(T_1, D), (T_2, D), ..., (T_B, D)]
    :param float pad_value: value to pad
    :return: padded batch with the shape (B, Tmax, D)
    :rtype: ndarray
    """
    import numpy as np
    bs = len(batch)
    maxlen = max([b.shape[0] for b in batch])
    if len(batch[0].shape) >= 2:
        batch_pad = np.zeros((bs, maxlen) + batch[0].shape[1:])
    else:
        batch_pad = np.zeros((bs, maxlen))
    batch_pad.fill(pad_value)
    for i, b in enumerate(batch):
        batch_pad[i, :b.shape[0]] = b
    del batch
    return batch_pad


# * -------------------- training iterator related -------------------- *


def make_batchset(data, batch_size, max_length_in, max_length_out, num_batches=0):
    # sort it by input lengths (long to short)
    sorted_data = sorted(data.items(), key=lambda data: int(
        data[1]['input'][0]['shape'][0]), reverse=True)
    logging.info('# utts: ' + str(len(sorted_data)))
    # change batchsize depending on the input and output length
    minibatch = []
    start = 0
    while True:
        ilen = int(sorted_data[start][1]['input'][0]['shape'][0])
        olen = int(sorted_data[start][1]['output'][0]['shape'][0])
        factor = max(int(ilen / max_length_in), int(olen / max_length_out))
        # if ilen = 1000 and max_length_in = 800
        # then b = batchsize / 2
        # and max(1, .) avoids batchsize = 0
        b = max(1, int(batch_size / (1 + factor)))
        end = min(len(sorted_data), start + b)
        minibatch.append(sorted_data[start:end])
        if end == len(sorted_data):
            break
        start = end
    if num_batches > 0:
        minibatch = minibatch[:num_batches]
    logging.info('# minibatches: ' + str(len(minibatch)))

    return minibatch

