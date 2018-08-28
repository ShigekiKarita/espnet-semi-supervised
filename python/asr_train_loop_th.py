#!/usr/bin/env python
import argparse
import collections
import contextlib
import copy
import json
import logging
import math
import os
import pickle
import random
import six

# spnet related
from e2e_asr_attctc_th import E2E
from e2e_asr_attctc_th import Loss
from asr_train_th import make_batchset, converter_kaldi, delete_feat
from results import EpochResult, GlobalResult

# third libaries
import lazy_io
import numpy as np
import torch


@contextlib.contextmanager
def open_kaldi_feat(batch, reader):
    try:
        yield converter_kaldi(batch, reader)
    finally:
        delete_feat(batch)


def get_parser():
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--gpu', '-g', default='-1', type=str,
                        help='GPU ID (negative value indicates CPU)')
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
    # TODO(karita): implement resume
    # parser.add_argument('--resume', '-r', default='',
    #                     help='Resume the training from snapshot')
    parser.add_argument('--minibatches', '-N', type=int, default='-1',
                        help='Process only N minibatches (for debug)')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    # task related
    parser.add_argument('--train-feat', type=str, required=True,
                        help='Filename of train feature data (Kaldi scp)')
    parser.add_argument('--valid-feat', type=str, required=True,
                        help='Filename of validation feature data (Kaldi scp)')
    parser.add_argument('--train-label', type=str, required=True,
                        help='Filename of train label data (json)')
    parser.add_argument('--valid-label', type=str, required=True,
                        help='Filename of validation label data (json)')
    # network archtecture
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
    # attention
    parser.add_argument('--atype', default='dot', type=str,
                        choices=['dot', 'location', 'noatt'],
                        help='Type of attention architecture')
    parser.add_argument('--adim', default=320, type=int,
                        help='Number of attention transformation dimensions')
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
    # minibatch related
    parser.add_argument('--batch-size', '-b', default=50, type=int,
                        help='Batch size')
    parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                        help='Batch size is reduced if the output sequence length > ML')
    # optimization related
    parser.add_argument('--opt', default='adadelta', type=str,
                        choices=['adadelta', 'adam'],
                        help='Optimizer')
    parser.add_argument('--lr', default=1.0, type=float,
                        help="Learning rate")
    parser.add_argument('--eps', default=1e-8, type=float,
                        help='Epsilon constant for optimizer')
    parser.add_argument('--eps-decay', default=0.01, type=float,
                        help='Decaying ratio of epsilon')
    parser.add_argument('--criterion', default='acc', type=str,
                        choices=['loss', 'acc'],
                        help='Criterion to perform epsilon decay')
    parser.add_argument('--threshold', default=1e-4, type=float,
                        help='Threshold to stop iteration')
    parser.add_argument('--epochs', '-e', default=30, type=int,
                        help='Number of maximum epochs')
    parser.add_argument('--grad-clip', default=5, type=float,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--supervised-data-ratio', default=1.0, type=float,
                        help='ratio of supervised training set')
    return parser


def setup_torch(args):
    # seed setting (chainer seed may not need it)
    nseed = args.seed
    random.seed(nseed)
    np.random.seed(nseed)
    torch.manual_seed(nseed)

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = True
        logging.info('pytorch cudnn deterministic is disabled')
    else:
        torch.backends.cudnn.deterministic = True

    # check cuda and cudnn availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')
    if not torch.backends.cudnn.enabled:
        logging.warning('cudnn is not available')


def setup(args):
    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    # display PYTHONPATH
    logging.info('python path = ' + os.environ['PYTHONPATH'])
    setup_torch(args)

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

    # get input and output dimension info
    with open(args.valid_label, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['idim'])
    odim = int(valid_json[utts[0]]['odim'])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.conf'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        # TODO(watanabe) use others than pickle, possibly json, and save as a text
        pickle.dump((idim, odim, args), f)
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # read json data
    with open(args.train_label, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_label, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    # that contains [ [{"utt-id": { "tokenid": ..., }} x batchsize], ... ]
    train_batch = make_batchset(train_json, args.batch_size,
                                args.maxlen_in, args.maxlen_out, args.minibatches)
    valid_batch = make_batchset(valid_json, args.batch_size,
                                args.maxlen_in, args.maxlen_out, args.minibatches)
    return idim, odim, train_batch, valid_batch


if __name__ == "__main__":
    args = get_parser().parse_args()
    idim, odim, train_batch, valid_batch = setup(args)
    if args.supervised_data_ratio != 1.0:
        n_supervised = int(len(train_batch) * args.supervised_data_ratio)
        train_batch = train_batch[:n_supervised]

    # specify model architecture
    e2e = E2E(idim, odim, args)
    model = Loss(e2e, args.mtlalpha)

    # Set gpu
    gpu_id = int(args.gpu)
    logging.info('gpu id: ' + str(gpu_id))
    if gpu_id >= 0:
        # Make a specified GPU current
        model.cuda(gpu_id)  # Copy the model to the GPU

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), lr=args.lr, rho=0.95, eps=args.eps)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # prepare Kaldi reader
    train_reader = lazy_io.read_dict_scp(args.train_feat)
    valid_reader = lazy_io.read_dict_scp(args.valid_feat)

    best = dict(loss=float("inf"), acc=-float("inf"))
    opt_key = "eps" if args.opt == "adadelta" else "lr"
    def get_opt_param():
        return optimizer.param_groups[0][opt_key]

    # training loop
    result = GlobalResult(args.epochs, args.outdir)
    for epoch in range(args.epochs):
        model.train()
        with result.epoch("main", train=True) as train_result:
            for batch in np.random.permutation(train_batch):
                with open_kaldi_feat(batch, train_reader) as x:
                    # forward
                    loss_ctc, loss_att, acc = model.predictor(x)
                    loss = args.mtlalpha * loss_ctc + (1 - args.mtlalpha) * loss_att
                    # backward
                    optimizer.zero_grad()  # Clear the parameter gradients
                    loss.backward()  # Backprop
                    loss.detach()  # Truncate the graph
                    # compute the gradient norm to check if it is normal or not
                    grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
                    logging.info('grad norm={}'.format(grad_norm))
                    if math.isnan(grad_norm):
                        logging.warning('grad norm is nan. Do not update model.')
                    else:
                        optimizer.step()
                    # print/plot stats to args.outdir/results
                    train_result.report({
                        "loss": loss,
                        "acc": acc,
                        "loss_ctc": loss_ctc,
                        "loss_att": loss_att,
                        "grad_norm": grad_norm,
                        opt_key: get_opt_param()
                    })

        with result.epoch("validation/main", train=False) as valid_result:
            model.eval()
            for batch in valid_batch:
                with open_kaldi_feat(batch, valid_reader) as x:
                    # forward (without backward)
                    loss_ctc, loss_att, acc = model.predictor(x)
                    loss = args.mtlalpha * loss_ctc + (1 - args.mtlalpha) * loss_att
                    # print/plot stats to args.outdir/results
                    valid_result.report({
                        "loss": loss,
                        "acc": acc,
                        "loss_ctc": loss_ctc,
                        "loss_att": loss_att,
                        opt_key: get_opt_param()
                    })

        # save/load model
        valid_avg = valid_result.average()
        degrade = False
        if best["loss"] > valid_avg["loss"]:
            best["loss"] = valid_avg["loss"]
            torch.save(model.state_dict(), args.outdir + "/model.loss.best")
        elif args.criterion == "loss":
            degrade = True

        if best["acc"] < valid_avg["acc"]:
            best["acc"] = valid_avg["acc"]
            torch.save(model.state_dict(), args.outdir + "/model.acc.best")
        elif args.criterion == "acc":
            degrade = True

        if degrade:
            key = "eps" if args.opt == "adadelta" else "lr"
            for p in optimizer.param_groups:
                p[key] *= args.eps_decay
            model.load_state_dict(torch.load(args.outdir + "/model." + args.criterion + ".best"))
