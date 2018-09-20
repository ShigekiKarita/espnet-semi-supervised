#!/usr/bin/env python

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import logging
import os
import platform
import random
import subprocess
import sys

from distutils.util import strtobool

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--gpu', default=None, type=int, nargs='?',
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--ngpu', default=0, type=int,
                        help='Number of GPUs')
    parser.add_argument('--backend', default='chainer', type=str,
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--dict', required=True,
                        help='Dictionary')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--debugdir', type=str,
                        help='Output directory for debugging')
    parser.add_argument('--resume', '-r', default='', nargs='?',
                        help='Resume the training from snapshot')
    parser.add_argument('--minibatches', '-N', type=int, default='-1',
                        help='Process only N minibatches (for debug)')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    # task related
    parser.add_argument('--train-feat', type=str, default=None,
                        help='Filename of train feature data (Kaldi scp)')
    parser.add_argument('--valid-feat', type=str, default=None,
                        help='Filename of validation feature data (Kaldi scp)')
    parser.add_argument('--train-json', type=str, default=None,
                        help='Filename of train label data (json)')
    parser.add_argument('--valid-json', type=str, default=None,
                        help='Filename of validation label data (json)')
    parser.add_argument('--train-label', type=str, default=None,
                        help='Filename of train label data (json)')
    parser.add_argument('--valid-label', type=str, default=None,
                        help='Filename of validation label data (json)')
    parser.add_argument('--train-utt2mode', type=str, default=None,
                        help='Filename of train mode data (scp)')
    parser.add_argument('--valid-utt2mode', type=str, default=None,
                        help='Filename of validation mode data (scp)')

    # ASR network archtecture
    # encoder
    parser.add_argument('--etype', default='blstmp', type=str,
                        choices=['blstm', 'blstmp', 'vggblstmp', 'vggblstm'],
                        help='Type of encoder network architecture')
    parser.add_argument('--elayers', default=4, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--eunits', '-u', default=300, type=int,
                        help='Number of encoder hidden units')
    parser.add_argument('--eprojs', default=320, type=int,
                        help='Number of encoder projection units')
    parser.add_argument('--subsample', default=1, type=str,
                        help='Subsample input frames x_y_z means subsample every x frame at 1st layer, '
                             'every y frame at 2nd layer etc.')
    # loss
    parser.add_argument('--ctc_type', default='warpctc', type=str,
                        choices=['chainer', 'warpctc'],
                        help='Type of CTC implementation to calculate loss.')
    # attention
    parser.add_argument('--atype', default='dot', type=str,
                        choices=['noatt', 'dot', 'add', 'location', 'coverage',
                                 'coverage_location', 'location2d', 'location_recurrent',
                                 'multi_head_dot', 'multi_head_add', 'multi_head_loc',
                                 'multi_head_multi_res_loc'],
                        help='Type of attention architecture')
    parser.add_argument('--adim', default=320, type=int,
                        help='Number of attention transformation dimensions')
    parser.add_argument('--awin', default=5, type=int,
                        help='Window size for location2d attention')
    parser.add_argument('--aheads', default=4, type=int,
                        help='Number of heads for multi head attention')
    parser.add_argument('--aconv-chans', default=-1, type=int,
                        help='Number of attention convolution channels \
                        (negative value indicates no location-aware attention)')
    parser.add_argument('--aconv-filts', default=100, type=int,
                        help='Number of attention convolution filters \
                        (negative value indicates no location-aware attention)')
    # decoder
    parser.add_argument('--dtype', default='lstm', type=str,
                        choices=['lstm'],
                        help='Type of decoder network architecture')
    parser.add_argument('--dlayers', default=1, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--dunits', default=320, type=int,
                        help='Number of decoder hidden units')
    parser.add_argument('--mtlalpha', default=0.5, type=float,
                        help='Multitask learning coefficient, alpha: alpha*ctc_loss + (1-alpha)*att_loss ')
    parser.add_argument('--lsm-type', const='', default='', type=str, nargs='?', choices=['', 'unigram'],
                        help='Apply label smoothing with a specified distribution type')
    parser.add_argument('--lsm-weight', default=0.0, type=float,
                        help='Label smoothing weight')
    # model (parameter) related
    parser.add_argument('--dropout-rate', default=0.0, type=float,
                        help='Dropout rate')
    # TTS network archtecture
    # encoder
    parser.add_argument('--tts-embed_dim', default=512, type=int,
                        help='Number of dimension of embedding')
    parser.add_argument('--tts-elayers', default=1, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--tts-eunits', default=512, type=int,
                        help='Number of encoder hidden units')
    parser.add_argument('--tts-econv_layers', default=3, type=int,
                        help='Number of encoder conv layers')
    parser.add_argument('--tts-econv_chans', default=512, type=int,
                        help='Number of encoder conv filter channels')
    parser.add_argument('--tts-econv_filts', default=5, type=int,
                        help='Number of encoder conv filter size')
    # attention
    parser.add_argument('--tts-adim', default=512, type=int,
                        help='Number of attention transformation dimensions')
    parser.add_argument('--tts-aconv-chans', default=32, type=int,
                        help='Number of attention convolution channels \
                        (negative value indicates no location-aware attention)')
    parser.add_argument('--tts-aconv-filts', default=32, type=int,
                        help='Number of attention convolution filters \
                        (negative value indicates no location-aware attention)')
    parser.add_argument('--tts-cumulate_att_w', default=True, type=strtobool,
                        help="Whether or not to cumulate attetion weights")
    # decoder
    parser.add_argument('--tts-dlayers', default=2, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--tts-dunits', default=1024, type=int,
                        help='Number of decoder hidden units')
    parser.add_argument('--tts-prenet_layers', default=2, type=int,
                        help='Number of prenet layers')
    parser.add_argument('--tts-prenet_units', default=256, type=int,
                        help='Number of prenet hidden units')
    parser.add_argument('--tts-postnet_layers', default=5, type=int,
                        help='Number of postnet layers')
    parser.add_argument('--tts-postnet_chans', default=512, type=int,
                        help='Number of postnet conv filter channels')
    parser.add_argument('--tts-postnet_filts', default=5, type=int,
                        help='Number of postnet conv filter size')
    parser.add_argument('--tts-output_activation', default=None, type=str, nargs='?',
                        help='Output activation function')
    # model (parameter) related
    parser.add_argument('--tts-use_speaker_embedding', default=False, type=strtobool,
                        help='Whether to use speaker embedding')
    parser.add_argument('--tts-use_batch_norm', default=True, type=strtobool,
                        help='Whether to use batch normalization')
    parser.add_argument('--tts-use_concate', default=True, type=strtobool,
                        help='Whether to concatenate encoder embedding with decoder outputs')
    parser.add_argument('--tts-use_residual', default=True, type=strtobool,
                        help='Whether to use residual connection in conv layer')
    parser.add_argument('--tts-dropout-rate', default=0.5, type=float,
                        help='Dropout rate')
    parser.add_argument('--tts-zoneout-rate', default=0.1, type=float,
                        help='Zoneout rate')
    # loss related
    parser.add_argument('--tts-monotonic', default=0.0, type=float,
                        help='Monotonic loss rate')
    parser.add_argument('--tts-use_masking', default=False, type=strtobool,
                        help='Whether to use masking in calculation of loss')
    parser.add_argument('--tts-bce_pos_weight', default=20.0, type=float,
                        help='Positive sample weight in BCE calculation (only for use_masking=True)')
    parser.add_argument('--asr_weight', default=0.1, type=float,
                        help='ASR loss weight (lr) in multi task loss')
    parser.add_argument('--tts_weight', default=1.0, type=float,
                        help='TTS loss weight (lr) in multi task loss')
    parser.add_argument('--s2s_weight', default=0.01, type=float,
                        help='S2S loss weight (lr) in multi task loss')
    parser.add_argument('--t2t_weight', default=0.01, type=float,
                        help='T2T loss weight (lr) in multi task loss')
    parser.add_argument('--mmd_weight', default=1.0, type=float,
                        help='Maximum mean discrepancy loss weight between encoded speech and text')
    # minibatch related
    parser.add_argument('--batch_sort_key', default=None, type=str,
                        choices=[None, 'output', 'input'], nargs='?',
                        help='Batch sorting key')
    parser.add_argument('--batch-size', '-b', default=50, type=int,
                        help='Batch size')
    parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                        help='Batch size is reduced if the output sequence length > ML')
    # optimization related
    parser.add_argument('--optim', default='Adam', type=str,
                        choices=['Adadelta', 'Adam'],
                        help='Optimizer')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate for optimizer')
    parser.add_argument('--eps', default=1e-8, type=float,
                        help='Epsilon constant for optimizer')
    parser.add_argument('--eps-decay', default=0.01, type=float,
                        help='Decaying ratio of epsilon')
    parser.add_argument('--weight-decay', default=1e-6, type=float,
                        help='Weight decay coefficient for optimizer')
    parser.add_argument('--criterion', default='acc', type=str,
                        choices=['loss', 'acc'],
                        help='Criterion to perform epsilon decay')
    parser.add_argument('--threshold', default=1e-4, type=float,
                        help='Threshold to stop iteration')
    parser.add_argument('--epochs', '-e', default=30, type=int,
                        help='Number of maximum epochs')
    parser.add_argument('--grad-clip', default=5, type=float,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--num-save-attention', default=3, type=int,
                        help='Number of samples of attention to be saved')
    # model road
    parser.add_argument('--model', default='', nargs='?',
                        help='Read ASR+TTTS model')
    parser.add_argument('--model-asr', default='', nargs='?',
                        help='Read ASR model')
    parser.add_argument('--model-tts', default='', nargs='?',
                        help='Read TTS model')
    args = parser.parse_args()

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    # check gpu argument
    if args.gpu is not None:
        logging.warn("--gpu option will be deprecated, please use --ngpu option.")
        if args.gpu == -1:
            args.ngpu = 0
        else:
            args.ngpu = 1

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        # python 2 case
        if platform.python_version_tuple()[0] == '2':
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]):
                cvd = subprocess.check_output(["/usr/local/bin/free-gpu", "-n", str(args.ngpu)]).strip()
                logging.info('CLSP: use gpu' + cvd)
                os.environ['CUDA_VISIBLE_DEVICES'] = cvd
        # python 3 case
        else:
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]).decode():
                cvd = subprocess.check_output(["/usr/local/bin/free-gpu", "-n", str(args.ngpu)]).decode().strip()
                logging.info('CLSP: use gpu' + cvd)
                os.environ['CUDA_VISIBLE_DEVICES'] = cvd

        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warn("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ['PYTHONPATH'])

    # set random seed
    logging.info('random seed = %d' % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dictionary for debug log
    if args.dict is not None:
        with open(args.dict, 'rb') as f:
            dictionary = f.readlines()
        char_list = [entry.decode('utf-8').split(' ')[0]
                     for entry in dictionary]
        char_list.insert(0, '<blank>')
        char_list.append('<eos>')
        args.char_list = char_list
    else:
        args.char_list = None

    # train
    logging.info('backend = ' + args.backend)
    if args.backend == "pytorch":
        from asrtts_pytorch import train
        train(args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
