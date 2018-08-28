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
from distutils.util import strtobool

# spnet related
from unsupervised import E2E, Discriminator
from e2e_asr_attctc_th import Loss
from asr_train_loop_th import get_parser, setup, open_kaldi_feat, make_batchset
from results import EpochResult, GlobalResult

# third libaries
import lazy_io
import numpy as np
import torch


def shuffle_pair(batch_list):
    keys = np.random.permutation([[b[0] for b in batch] for batch in batch_list])
    vals = np.random.permutation([[b[1] for b in batch] for batch in batch_list])
    ret = []
    for ks, vs in zip(keys, vals):
        batch = []
        n = min(len(ks), len(vs))
        for k, v in zip(ks[:n], vs[:n]):
            batch.append((k, v))
        ret.append(batch)
    return ret

def cpu_loader(storage, location):
    return storage

def load_pretrained(self, src_dict, idim, odim, args, train_batch, train_reader):
    dst_dict = self.state_dict()
    for k, v in src_dict.items():
        assert k in dst_dict, k + " not found"
        dst_dict[k] = v
    self.load_state_dict(dst_dict)
    tgt_dict = self.state_dict()
    for k, v in src_dict.items():
        assert (tgt_dict[k] == v).all()

    if args.verbose > 0:
        import e2e_asr_attctc_th as base
        init = base.Loss(base.E2E(idim, odim, args), args.mtlalpha)
        init.load_state_dict(src_dict)
        init.eval()
        self.predictor.eval()
        # test first batch prediction equality
        with open_kaldi_feat(train_batch[0], train_reader) as data:
            init_ctc, init_att, init_acc = init.predictor(data)
            re_ctc, re_att, re_acc = self.predictor(data, supervised=True)
        print("init: ", init_ctc, init_att, init_acc)
        print("re:   ", re_ctc, re_att, re_acc)
        np.testing.assert_almost_equal(init_ctc.data[0], re_ctc.data[0])
        np.testing.assert_almost_equal(init_att.data[0], re_att.data[0])
        np.testing.assert_almost_equal(init_acc, re_acc)
    return self


def parameters(model, exclude=None):
    if exclude is None:
        return model.parameters()
    assert exclude in model.modules()
    exclude_params = list(exclude.parameters())
    model_params = list(model.parameters())
    ret = []
    for p in model_params:
        found = False
        for e in exclude_params:
            if p is e:
                found = True
                break
        if not found:
            ret.append(p)
    assert len(ret) == (len(model_params) - len(exclude_params))
    return ret


def fully_unpaired(batch_list):
    """
    insert values of next batch into prev batch (make key-value mismatch)
    """
    ret = []
    for i, batch in enumerate(batch_list):
        ret_batch = []
        if i % 2 == 0 and i != len(batch_list) - 1:
            next_batch = batch_list[i+1]
            ret.append([(k1, v2) for (k1, v1), (k2, v2) in zip(batch, next_batch)])
    return ret


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--init-model", type=str, default="None")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--discriminator-dim", type=int, default=320)
    parser.add_argument("--unsupervised-feat", type=str)
    parser.add_argument("--unsupervised-json", type=str)
    parser.add_argument('--speech-text-ratio', default=0.5, type=float,
                        help='Multitask learning coefficient')
    parser.add_argument('--supervised-loss-ratio', default=0.9, type=float,
                        help='Multitask learning coefficient')
    parser.add_argument('--unsupervised-loss', choices=["None", "gan", "gauss", "gausslogdet", "variance", "mmd"], default="None", type=str,
                        help='loss for hidden space')
    parser.add_argument('--use-batchnorm', default=False, type=strtobool,
                        help="use batchnorm in output of encoder")
    parser.add_argument('--use-smaller-data-size', default=True, type=strtobool,
                        help="use smaller size of supervised/unsupervised dataset for iteration")
    parser.add_argument('--lock-encoder', default=False, type=strtobool,
                        help="do not update encoder parameters")

    args = parser.parse_args()
    idim, odim, supervised_train_batch, valid_batch = setup(args)
    if args.supervised_data_ratio != 1.0:
        n_supervised = int(len(supervised_train_batch) * args.supervised_data_ratio)
        supervised_train_batch = supervised_train_batch[:n_supervised]

    with open(args.unsupervised_json, 'rb') as f:
        unsupervised_json = json.load(f)['utts']
    unsupervised_train_batch = make_batchset(unsupervised_json, args.batch_size,
                                             args.maxlen_in, args.maxlen_out, args.minibatches)
    unsupervised_train_batch = fully_unpaired(unsupervised_train_batch)

    n_supervised = len(supervised_train_batch)
    n_unsupervised = len(unsupervised_train_batch)

    # prepare Kaldi reader
    train_reader = lazy_io.read_dict_scp(args.train_feat)
    valid_reader = lazy_io.read_dict_scp(args.valid_feat)
    unsupervised_reader = lazy_io.read_dict_scp(args.unsupervised_feat)

    # specify model architecture
    e2e = E2E(idim, odim, args)
    model = Loss(e2e, args.mtlalpha)
    if args.init_model != "None":
        src_dict = torch.load(args.init_model, map_location=cpu_loader)
        model = load_pretrained(model, src_dict, idim, odim, args, supervised_train_batch, train_reader)
    if args.unsupervised_loss == "gan":
        discriminator = Discriminator(args.eprojs, args.discriminator_dim)
    else:
        discriminator = None

    # Set gpu
    gpu_id = int(args.gpu)
    logging.info('gpu id: ' + str(gpu_id))
    if gpu_id >= 0:
        # Make a specified GPU current
        model.cuda(gpu_id)  # Copy the model to the GPU
        if discriminator:
            discriminator.cuda(gpu_id)

    # Setup an optimizer
    if args.lock_encoder:
        model_params = parameters(model, model.predictor.enc)
    else:
        model_params = model.parameters()
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model_params, lr=args.lr, rho=0.95, eps=args.eps, weight_decay=args.weight_decay)
        if discriminator:
            d_optimizer = torch.optim.Adadelta(
                discriminator.parameters(), lr=args.lr, rho=0.95, eps=args.eps, weight_decay=args.weight_decay)

    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model_params, lr=args.lr, weight_decay=args.weight_decay)
        if discriminator:
            d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = dict(loss=float("inf"), acc=-float("inf"))
    opt_key = "eps" if args.opt == "adadelta" else "lr"
    def get_opt_param():
        return optimizer.param_groups[0][opt_key]

    # training loop
    result = GlobalResult(args.epochs, args.outdir)
    for epoch in range(args.epochs):
        model.train()
        with result.epoch("main", train=True) as train_result:
            n_iter_fun = min if args.use_smaller_data_size else max
            for i in range(n_iter_fun(n_supervised, n_unsupervised)):
                # re-shuffle and repeat smaller one
                if i % n_supervised == 0:
                    supervised_train_batch = np.random.permutation(supervised_train_batch)
                if i % n_unsupervised == 0:
                    unsupervised_train_batch = shuffle_pair(unsupervised_train_batch)
                sbatch = supervised_train_batch[i % n_supervised]
                ubatch = unsupervised_train_batch[i % n_unsupervised]

                # supervised forward
                with open_kaldi_feat(sbatch, train_reader) as sx:
                    loss_ctc, loss_att, acc_supervised = model.predictor(sx, supervised=True)
                loss_supervised = args.mtlalpha * loss_ctc + (1.0 - args.mtlalpha) * loss_att

                # unsupervised forward
                with open_kaldi_feat(ubatch, unsupervised_reader) as ux:
                    loss_text, loss_hidden, acc_text = model.predictor(ux, supervised=False, discriminator=discriminator)

                loss_unsupervised = args.speech_text_ratio * loss_hidden + (1.0 - args.speech_text_ratio) * loss_text

                if discriminator:
                    loss_discriminator = -loss_hidden
                    d_optimizer.zero_grad()
                    loss_discriminator.backward(retain_variables=True)
                    loss_discriminator.detach()
                    d_optimizer.step()

                optimizer.zero_grad()  # Clear the parameter gradients
                loss = args.supervised_loss_ratio * loss_supervised + (1.0 - args.supervised_loss_ratio) * loss_unsupervised
                loss.backward()
                loss.detach()

                # compute the gradient norm to check if it is normal or not
                grad_norm = torch.nn.utils.clip_grad_norm(model_params, args.grad_clip)
                logging.info('grad norm={}'.format(grad_norm))
                if math.isnan(grad_norm):
                    logging.warning('grad norm is nan. Do not update model.')
                else:
                    optimizer.step()
                optimizer.zero_grad()  # Clear the parameter gradients

                # print/plot stats to args.outdir/results
                results = {
                    "loss_att": loss_att.data[0],
                    "loss_ctc": loss_ctc,
                    "loss_supervised": loss_supervised.data[0],
                    "acc_supervised": acc_supervised,

                    "loss": loss.data[0],
                    "loss_unsupervised": loss_unsupervised.data[0] / (1.0 - args.supervised_loss_ratio),
                    "acc_text": acc_text,
                    "loss_text": loss_text.data[0],
                    "loss_hidden": loss_hidden,

                    "grad_norm": 0.0 if math.isnan(grad_norm) else grad_norm,
                    opt_key: get_opt_param()
                }
                # print(acc_supervised)
                # if not math.isnan(grad_norm):
                train_result.report(results)

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

