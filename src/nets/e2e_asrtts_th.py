#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import logging
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from e2e_asr_th import to_cuda
from e2e_asr_th import pad_list
from e2e_tts_th import make_mask

from asrtts_utils import pad_ndarray_list


torch_is_old = torch.__version__.startswith("0.3.")


def mmd_loss(xs,ys,beta=1.0):
    Nx = xs.shape[0]
    Ny = ys.shape[0]
    Kxy = torch.matmul(xs,ys.t())
    dia1 = torch.sum(xs*xs,1)
    dia2 = torch.sum(ys*ys,1)
    Kxy = Kxy-0.5*dia1.unsqueeze(1).expand(Nx,Ny)
    Kxy = Kxy-0.5*dia2.expand(Nx,Ny)
    Kxy = torch.exp(beta*Kxy).sum()/Nx/Ny

    Kx = torch.matmul(xs,xs.t())
    Kx = Kx-0.5*dia1.unsqueeze(1).expand(Nx,Nx)
    Kx = Kx-0.5*dia1.expand(Nx,Nx)
    Kx = torch.exp(beta*Kx).sum()/Nx/Nx

    Ky = torch.matmul(ys,ys.t())
    Ky = Ky-0.5*dia2.unsqueeze(1).expand(Ny,Ny)
    Ky = Ky-0.5*dia2.expand(Ny,Ny)
    Ky = torch.exp(beta*Ky).sum()/Ny/Ny
    return Kx+Ky-2*Kxy


def packed_mmd(hspad, hslens, htpad, htlens):
    from torch.nn.utils.rnn import pack_padded_sequence
    hspack = pack_padded_sequence(hspad, hslens, batch_first=True)
    htpack = pack_padded_sequence(htpad, htlens, batch_first=True)
    return mmd_loss(hspack.data, hspack.data)


class ASRTTSLoss(torch.nn.Module):
    def __init__(self, asr_loss, tts_loss, args, return_targets=True):
        super(ASRTTSLoss, self).__init__()
        self.asr_loss = asr_loss
        self.tts_loss = tts_loss

        self.ae_speech = AutoEncoderSpeech(asr_loss, tts_loss, args, tts_loss.use_masking, tts_loss.bce_pos_weight,
                                           use_speaker_embedding=self.tts_loss.model.spk_embed_dim)
        self.ae_text = AutoEncoderText(asr_loss, tts_loss, args, use_speaker_embedding=self.tts_loss.model.spk_embed_dim)

        self.return_targets = return_targets
        self.mmd_weight = args.mmd_weight
        self.return_hidden = args.mmd_weight != 0.0


def get_asr_data(self, data, sort_by):
    # utt list of frame x dim
    xs = [d[1]['feat_asr'] for d in data]
    # remove 0-output-length utterances
    tids = [d[1]['output'][0]['tokenid'].split() for d in data]
    filtered_index = filter(lambda i: len(tids[i]) > 0, range(len(xs)))
    if sort_by == 'feat':
        sorted_index = sorted(filtered_index, key=lambda i: -len(xs[i]))
    elif sort_by == 'text':
        sorted_index = sorted(filtered_index, key=lambda i: -len(tids[i]))
    else:
        logging.error("Error: specify 'text' or 'feat' to sort")
        sys.exit()
    if len(sorted_index) != len(xs):
        logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
            len(xs), len(sorted_index)))
    xs = [xs[i] for i in sorted_index]
    # utt list of olen
    texts = [np.fromiter(map(int, tids[i]), dtype=np.int64) for i in sorted_index]
    if torch_is_old:
        texts = [to_cuda(self, Variable(torch.from_numpy(y), volatile=not self.training)) for y in texts]
    else:
        texts = [to_cuda(self, torch.from_numpy(y)) for y in texts]

    # subsample frame
    xs = [xx[::self.subsample[0], :] for xx in xs]
    featlens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
    if torch_is_old:
        hs = [to_cuda(self, Variable(torch.from_numpy(xx), volatile=not self.training)) for xx in xs]
    else:
        hs = [to_cuda(self, torch.from_numpy(xx)) for xx in xs]

    # 1. encoder
    feats = pad_list(hs, 0.0)

    return texts, feats, featlens


def get_tts_data(self, data, sort_by, use_speaker_embedding=None):
    # get eos
    eos = str(int(data[0][1]['output'][0]['shape'][1]) - 1)

    # get target features and input character sequence
    texts = [b[1]['output'][0]['tokenid'].split() + [eos] for b in data]
    feats = [b[1]['feat_tts'] for b in data]

    # remove empty sequence and get sort along with length
    filtered_idx = filter(lambda i: len(texts[i]) > 0, range(len(feats)))
    if sort_by == 'feat':
        sorted_idx = sorted(filtered_idx, key=lambda i: -len(feats[i]))
    elif sort_by == 'text':
        sorted_idx = sorted(filtered_idx, key=lambda i: -len(texts[i]))
    else:
        logging.error("Error: specify 'text' or 'feat' to sort")
        sys.exit()
    texts = [np.fromiter(map(int, texts[i]), dtype=np.int64) for i in sorted_idx]
    feats = [feats[i] for i in sorted_idx]

    # get list of lengths (must be tensor for DataParallel)
    textlens = torch.from_numpy(np.fromiter((x.shape[0] for x in texts), dtype=np.int64))
    featlens = torch.from_numpy(np.fromiter((y.shape[0] for y in feats), dtype=np.int64))

    # perform padding and convert to tensor
    texts = torch.from_numpy(pad_ndarray_list(texts, 0)).long()
    feats = torch.from_numpy(pad_ndarray_list(feats, 0)).float()

    # make labels for stop prediction
    labels = feats.new(feats.size(0), feats.size(1)).zero_()
    for i, l in enumerate(featlens):
        labels[i, l - 1:] = 1

    if torch_is_old:
        texts = to_cuda(self, texts, volatile=not self.training)
        feats = to_cuda(self, feats, volatile=not self.training)
        labels = to_cuda(self, labels, volatile=not self.training)
    else:
        texts = to_cuda(self, texts)
        feats = to_cuda(self, feats)
        labels = to_cuda(self, labels)

    # load speaker embedding
    if use_speaker_embedding is not None:
        spembs = [b[1]['feat_spembs'] for b in data]
        spembs = [spembs[i] for i in sorted_idx]
        spembs = torch.from_numpy(np.array(spembs)).float()

        if torch_is_old:
            spembs = to_cuda(self, spembs, volatile=not self.training)
        else:
            spembs = to_cuda(self, spembs)
    else:
        spembs = None

    if self.return_targets:
        return texts, textlens, feats, labels, featlens, spembs
    else:
        return texts, textlens, feats, spembs


def get_subsample(args):
    subsample = np.ones(args.elayers + 1, dtype=np.int)
    if args.etype == 'blstmp':
        ss = args.subsample.split("_")
        for j in range(min(args.elayers + 1, len(ss))):
            subsample[j] = int(ss[j])
    else:
        logging.warning('Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
    logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))

    return subsample


class AutoEncoderSpeech(torch.nn.Module):
    def __init__(self, asr_loss, tts_loss, args, use_masking, bce_pos_weight,
                 return_targets=True, use_speaker_embedding=None):
        super(AutoEncoderSpeech, self).__init__()
        self.asr_enc = asr_loss.predictor.enc
        self.tts_dec = tts_loss.model.dec
        self.subsample = get_subsample(args)
        self.use_masking = use_masking
        self.bce_pos_weight = bce_pos_weight
        self.return_targets = return_targets
        self.use_speaker_embedding = use_speaker_embedding

    def forward(self, data, return_hidden=False):
        asr_texts, asr_feats, asr_featlens = get_asr_data(self, data, 'feat')
        tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens, spembs = \
            get_tts_data(self, data, 'feat', self.use_speaker_embedding)

        # encoder
        hpad_pre_spk, hlens = self.asr_enc(asr_feats, asr_featlens)
        if self.use_speaker_embedding is not None:
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hpad_pre_spk.size(1), -1)
            hpad = torch.cat([hpad_pre_spk, spembs], dim=-1)
        else:
            hpad = hpad_pre_spk

        after_outs, before_outs, logits, att_ws = self.tts_dec(hpad, hlens.tolist(), tts_feats)
        # copied from e2e_tts_th.py
        if self.use_masking and tts_featlens is not None:
            # weight positive samples
            if self.bce_pos_weight != 1.0:
                # TODO(kan-bayashi): need to be fixed in pytorch v4
                weights = tts_feats.data.new(*tts_labels.size()).fill_(1)
                if torch_is_old:
                    weights = Variable(weights, volatile=tts_feats.volatile)
                weights.masked_fill_(tts_labels.eq(1), self.bce_pos_weight)
            else:
                weights = None
            # masking padded values
            mask = to_cuda(self, make_mask(tts_featlens, tts_feats.size(2)))
            feats = tts_feats.masked_select(mask)
            after_outs = after_outs.masked_select(mask)
            before_outs = before_outs.masked_select(mask)
            labels = tts_labels.masked_select(mask[:, :, 0])
            logits = logits.masked_select(mask[:, :, 0])
            weights = weights.masked_select(mask[:, :, 0]) if weights is not None else None
            # calculate loss
            l1_loss = F.l1_loss(after_outs, feats) + F.l1_loss(before_outs, feats)
            mse_loss = F.mse_loss(after_outs, feats) + F.mse_loss(before_outs, feats)
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels, weights)
            loss = l1_loss + mse_loss + bce_loss
        else:
            # calculate loss
            l1_loss = F.l1_loss(after_outs, tts_feats) + F.l1_loss(before_outs, tts_feats)
            mse_loss = F.mse_loss(after_outs, tts_feats) + F.mse_loss(before_outs, tts_feats)
            bce_loss = F.binary_cross_entropy_with_logits(logits, tts_labels)
            loss = l1_loss + mse_loss + bce_loss

        # report loss values for logging
        loss_data = loss.data[0] if torch_is_old else loss.item()
        l1_loss_data = l1_loss.data[0] if torch_is_old else l1_loss.item()
        bce_loss_data = bce_loss.data[0] if torch_is_old else bce_loss.item()
        mse_loss_data = mse_loss.data[0] if torch_is_old else mse_loss.item()
        logging.debug("loss = %.3e (bce: %.3e, l1: %.3e, mse: %.3e)" % (
            loss_data, bce_loss_data, l1_loss_data, mse_loss_data))

        if return_hidden:
            return loss, hpad_pre_spk, hlens
        return loss


class AutoEncoderText(torch.nn.Module):
    def __init__(self, asr_loss, tts_loss, args, return_targets=True, use_speaker_embedding=None):
        super(AutoEncoderText, self).__init__()
        self.tts_enc = tts_loss.model.enc
        self.asr_dec = asr_loss.predictor.dec
        self.subsample = get_subsample(args)
        self.return_targets = return_targets
        self.use_speaker_embedding = use_speaker_embedding

    def forward(self, data, return_hidden=False):
        asr_texts, asr_feats, asr_featlens = get_asr_data(self, data, 'text')
        tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens, spembs = \
            get_tts_data(self, data, 'text', self.use_speaker_embedding)
        
        if isinstance(tts_textlens, torch.Tensor) or isinstance(tts_textlens, np.ndarray):
            tts_textlens = list(map(int, tts_textlens))

        hpad, hlens = self.tts_enc(tts_texts, tts_textlens)

        # NOTE asr_texts and tts_texts would be different due to the <eos> treatment
        loss, acc = self.asr_dec(hpad, hlens, asr_texts)

        if return_hidden:
            return loss, acc, hpad, hlens
        return loss, acc
