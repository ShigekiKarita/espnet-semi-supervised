#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
import json
import logging
import math
import numpy as np
import random
import os
import pickle
import sys

# chainer related
import chainer

from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions
from chainer.training import extension

# torch related
import torch
from torch.autograd import Variable

# spnet related
from asr_utils import adadelta_eps_decay
from asr_utils import CompareValueTrigger
from asr_utils import load_labeldict
from asr_utils import make_batchset
from asr_utils import restore_snapshot

# from tts_pytorch import pad_ndarray_list

from e2e_asr_th import E2E
from e2e_asr_th import Loss
from e2e_asr_th import pad_list

torch_is_old = False

from e2e_tts_th import Tacotron2
from e2e_tts_th import Tacotron2Loss

from e2e_asrtts_th import packed_mmd
from e2e_asrtts_th import ASRTTSLoss
from e2e_asrtts_th import get_asr_data
from e2e_asrtts_th import get_tts_data

from asrtts_utils import pad_ndarray_list

# for kaldi io
import kaldi_io_py

# matplotlib related
import matplotlib
matplotlib.use('Agg')

# some tuning params
REPORT_INTERVAL = 100
ALL_MODE = True # False
FREEZE_ATT = True
NO_AVG = True
MODEL_FULL = False


from asr_utils import PlotAttentionReport
class CustomPlotAttentionReport(PlotAttentionReport):
    def __init__(self, model, isASR, data, outdir, reverse=False):
        self.isASR = isASR
        if hasattr(model, "module"):
            self.model = model.module
        else:
            self.model = model
        if isASR:
            att_vis_fn = self.model.asr_loss.predictor.calculate_all_attentions
        else:
            att_vis_fn = self.model.tts_loss.model.calculate_all_attentions

        converter = None
        device = None
        super().__init__(att_vis_fn, data, outdir, converter, device, reverse)

    def __call__(self, trainer):
        with torch.no_grad():
            if self.isASR:
                asr_texts, asr_feats, asr_featlens = get_asr_torch(self.model, self.data)
                att_ws = self.att_vis_fn(asr_feats, asr_featlens, asr_texts)
            else:
                tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens, spembs \
                    = get_tts_torch(self.model, self.data)
                att_ws = self.att_vis_fn(tts_texts, tts_textlens, tts_feats, spembs)

        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.ep.{.updater.epoch}.png" % (
                self.outdir, self.data[idx][0])
            if self.reverse:
                dec_len = int(self.data[idx][1]['input'][0]['shape'][0])
                enc_len = int(self.data[idx][1]['output'][0]['shape'][0])
            else:
                dec_len = int(self.data[idx][1]['output'][0]['shape'][0])
                enc_len = int(self.data[idx][1]['input'][0]['shape'][0])
            if len(att_w.shape) == 3:
                att_w = att_w[:, :dec_len, :enc_len]
            else:
                att_w = att_w[:dec_len, :enc_len]
            self._plot_and_save_attention(att_w, filename.format(trainer))


def converter_kaldi(batch, device=None, use_speaker_embedding=None):
    # batch only has one minibatch utterance, which is specified by batch[0]
    batch = batch[0]
    for data in batch:
        feat_asr = kaldi_io_py.read_mat(data[1]['input'][0]['feat'])
        feat_tts = kaldi_io_py.read_mat(data[1]['input'][1]['feat'])
        data[1]['feat_asr'] = feat_asr
        data[1]['feat_tts'] = feat_tts
        if use_speaker_embedding is not None:
            feat_spembs = kaldi_io_py.read_vec_flt(data[1]['input'][2]['feat'])
            data[1]['feat_spembs'] = feat_spembs

    return batch


def delete_feat(batch):
    for data in batch:
        del data[1]['feat_asr']
        del data[1]['feat_tts']
        if 'feat_spembs' in data[1]:
            del data[1]['feat_spembs']

    return batch


def get_asr_torch(model, data):
    asr_texts, asr_feats, asr_featlens = get_asr_data(model.asr_loss.predictor, data, 'feat')
    asr_texts = pad_list(asr_texts, model.asr_loss.predictor.dec.ignore_id)
    return asr_texts, asr_feats, asr_featlens

def get_tts_torch(model, data):
    return get_tts_data(model, data, 'text', use_speaker_embedding=model.tts_loss.model.spk_embed_dim)


class CustomEvaluater(extensions.Evaluator):
    '''Custom evaluater for pytorch'''

    def __init__(self, model, iterator, target, converter, device):
        super(CustomEvaluater, self).__init__(iterator, target, converter=converter, device=device)
        self.model = model

    # The core part of the update routine can be customized by overriding.
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        if not torch_is_old:
            torch.set_grad_enabled(False)
        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                # read scp files
                # x: original json with loaded features
                #    will be converted to chainer variable later
                data = self.converter(batch, use_speaker_embedding=self.model.tts_loss.model.spk_embed_dim)
                self.model.eval()
                if data[0][1]['utt2mode'] != 'p':
                    logging.error("Error: evaluation only support a parallel data mode ('p')")
                    sys.exit()

                asr_texts, asr_feats, asr_featlens = get_asr_torch(self.model, data)
                asr_loss, asr_acc = self.model.asr_loss(asr_feats, asr_featlens, asr_texts,
                                                        do_report=False, report_acc=True)  # disable reporter

                tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens, spembs = \
                    get_tts_data(self.model, data, 'text', use_speaker_embedding=self.model.tts_loss.model.spk_embed_dim)
                avg_textlen = float(np.mean(tts_textlens.data.cpu().numpy()))
                tts_loss = self.model.tts_loss(tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens,
                                               spembs=spembs, do_report=False)
                s2s_loss = self.model.ae_speech(data)
                t2t_loss, t2t_acc = self.model.ae_text(data)

                # average loss for all four networks
                if NO_AVG:
                    loss = (asr_loss + tts_loss + s2s_loss + t2t_loss)/ 4.0
                else:
                    loss = (asr_loss/avg_textlen + tts_loss + s2s_loss + t2t_loss/avg_textlen)/ 4.0
                loss_data = loss.item() if torch_is_old else loss.item()
                if NO_AVG:
                    asr_loss_data = asr_loss.item() if torch_is_old else asr_loss.item()
                else:
                    asr_loss_data = asr_loss.item()/avg_textlen if torch_is_old else asr_loss.item()/avg_textlen
                tts_loss_data = tts_loss.item() if torch_is_old else tts_loss.item()
                s2s_loss_data = s2s_loss.item() if torch_is_old else s2s_loss.item()
                if NO_AVG:
                    t2t_loss_data = t2t_loss.item() if torch_is_old else t2t_loss.item()
                else:
                    t2t_loss_data = t2t_loss.item()/avg_textlen if torch_is_old else t2t_loss.item()/avg_textlen

                chainer.reporter.report({'d/loss': loss_data})
                chainer.reporter.report({'d/asr_loss': asr_loss_data})
                chainer.reporter.report({'d/tts_loss': tts_loss_data})
                chainer.reporter.report({'d/asr_acc': asr_acc})
                chainer.reporter.report({'d/s2s_loss': s2s_loss_data})
                chainer.reporter.report({'d/t2t_loss': t2t_loss_data})
                chainer.reporter.report({'d/t2t_acc': t2t_acc})

                delete_feat(data)

            summary.add(observation)

        if not torch_is_old:
            torch.set_grad_enabled(True)
        self.model.train()

        return summary.compute_mean()


def update_parameters(att):
    for child in att.children():
        for param in child.parameters():
            if param.grad is not None:
                param.grad.data.zero_()


class CustomUpdater(training.StandardUpdater):
    '''Custom updater for pytorch'''

    def __init__(self, model, grad_clip_threshold, train_iter,
                 opts, converter, device, args):
        super(CustomUpdater, self).__init__(
            train_iter, opts, converter=converter, device=None)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.num_gpu = len(device)
        self.opts = opts
        self.clip_grad_norm = torch.nn.utils.clip_grad_norm_
        self.args = args

    def gradient_decent(self, loss, optimizer, freeze_att=False, retain_graph=False):
        optimizer.zero_grad()  # Clear the parameter gradients
        loss.backward(retain_graph=retain_graph)  # Backprop
        loss.detach()  # Truncate the graph

        if freeze_att:
            update_parameters(self.model.tts_loss.model.dec.att)
            update_parameters(self.model.asr_loss.predictor.att)

        # compute the gradient norm to check if it is normal or not
        grad_norm = self.clip_grad_norm(self.model.parameters(), self.grad_clip_threshold)
        logging.info('total grad norm={}'.format(grad_norm))
        grad_norm = self.clip_grad_norm(self.model.asr_loss.parameters(), self.grad_clip_threshold)
        logging.info('asr grad norm={}'.format(grad_norm))
        grad_norm = self.clip_grad_norm(self.model.tts_loss.parameters(), self.grad_clip_threshold)
        logging.info('tts grad norm={}'.format(grad_norm))
        grad_norm = self.clip_grad_norm(self.model.ae_speech.asr_enc.parameters(), self.grad_clip_threshold)
        grad_norm += self.clip_grad_norm(self.model.ae_speech.tts_dec.parameters(), self.grad_clip_threshold)
        logging.info('s2s grad norm={}'.format(grad_norm))
        grad_norm = self.clip_grad_norm(self.model.ae_text.tts_enc.parameters(), self.grad_clip_threshold)
        grad_norm += self.clip_grad_norm(self.model.ae_text.asr_dec.parameters(), self.grad_clip_threshold)
        logging.info('t2t grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.__next__()

        # read scp files
        # x: original json with loaded features
        #    will be converted to chainer variable later
        # batch only has one minibatch utterance, which is specified by batch[0]
        if len(batch[0]) < self.num_gpu:
            logging.warning('batch size is less than number of gpus. Ignored')
            return
        data = self.converter(batch, use_speaker_embedding=self.model.tts_loss.model.spk_embed_dim)

        # Compute the loss at this time step and accumulate it
        asr_texts, asr_feats, asr_featlens = get_asr_torch(self.model, data)
        tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens, spembs = \
            get_tts_data(self.model, data, 'text', use_speaker_embedding=self.model.tts_loss.model.spk_embed_dim)

        avg_textlen = float(np.mean(tts_textlens.data.cpu().numpy()))
        if data[0][1]['utt2mode'] == 'p':
            logging.info("parallel data mode")
            if ALL_MODE:
                modes = ['asr', 'tts', 's2s', 't2t']
            else:
                modes = ['asr', 'tts']
            random.shuffle(modes)
            # shuffle
            loss_data_sum = 0.0
            use_mmd = self.args.mmd_weight != 0.0
            for mode in modes:
                if mode == 'asr':
                    if self.args.asr_weight == 0.0:
                        asr_loss_data = 0  # use nan?
                        asr_acc = 0
                        continue
                    loss, asr_acc = self.model.asr_loss(asr_feats, asr_featlens, asr_texts,
                                                        do_report=False, report_acc=True)  # disable reporter
                    if NO_AVG:
                        pass
                    else:
                        loss = loss/avg_textlen
                    asr_loss_data = loss.item()
                    chainer.reporter.report({'t/asr_loss': asr_loss_data})
                    chainer.reporter.report({'t/asr_acc': asr_acc})
                    loss_data_sum += asr_loss_data
                    logging.info("asr_loss_data: %f", asr_loss_data)
                    self.gradient_decent(loss, self.opts[mode])
                if mode == 'tts':
                    if self.args.tts_weight == 0.0:
                        tts_loss_data = 0  # use nan?
                        continue

                    loss = self.model.tts_loss(tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens,
                                               spembs=spembs, do_report=False)
                    tts_loss_data = loss.item()
                    chainer.reporter.report({'t/tts_loss': tts_loss_data})
                    loss_data_sum += tts_loss_data
                    logging.info("tts_loss_data: %f", tts_loss_data)
                    self.gradient_decent(loss, self.opts[mode])
                if mode == 's2s':
                    if self.args.s2s_weight == 0.0 and not use_mmd:
                        s2s_loss_data = 0.0
                        continue
                    loss, hspad, hslen = self.model.ae_speech(data, return_hidden=True)
                    s2s_loss_data = loss.item()
                    chainer.reporter.report({'t/s2s_loss': s2s_loss_data})
                    loss_data_sum += s2s_loss_data
                    logging.info("s2s_loss_data: %f", s2s_loss_data)
                    if self.args.s2s_weight != 0.0:
                        self.gradient_decent(loss, self.opts[mode], freeze_att=FREEZE_ATT, retain_graph=use_mmd)
                if mode == 't2t':
                    if self.args.t2t_weight == 0.0 and not use_mmd:
                        t2t_loss_data = 0.0
                        continue
                    loss, t2t_acc, htpad, htlen = self.model.ae_text(data, return_hidden=True)
                    if NO_AVG:
                        pass
                    else:
                        loss = loss/avg_textlen
                    t2t_loss_data = loss.item()
                    chainer.reporter.report({'t/t2t_loss': t2t_loss_data})
                    chainer.reporter.report({'t/t2t_acc': t2t_acc})
                    loss_data_sum += t2t_loss_data
                    logging.info("t2t_loss_data: %f", t2t_loss_data)
                    if self.args.t2t_weight != 0.0:
                        self.gradient_decent(loss, self.opts[mode], freeze_att=FREEZE_ATT, retain_graph=use_mmd)
                logging.info("loss_data_sum: %f", loss_data_sum)

            if use_mmd:
                loss = packed_mmd(hspad, hslen, htpad, htlen)
                chainer.reporter.report({'t/mmd_loss': loss.item() })
                self.gradient_decent(loss, self.opts["mmd"], freeze_att=FREEZE_ATT)
                loss_data = (loss_data_sum + loss.item()) / 5.0
            elif ALL_MODE:
                loss_data = loss_data_sum / 4.0
            else:
                loss_data = loss_data_sum / 2.0

            logging.info("loss_data: %f", loss_data)
            chainer.reporter.report({'t/loss': loss_data})
        elif data[0][1]['utt2mode'] == 'a':
            logging.info("audio only mode")
            s2s_loss = self.model.ae_speech(data)
            loss = s2s_loss
            self.gradient_decent(loss, self.opts['s2s'], freeze_att=FREEZE_ATT)

            loss_data = loss.item()
            logging.info("loss: %f", loss_data)
            chainer.reporter.report({'t/loss': loss_data})
            chainer.reporter.report({'t/s2s_loss': loss_data})
        elif data[0][1]['utt2mode'] == 't':
            logging.info("text only mode")
            t2t_loss, t2t_acc = self.model.ae_text(data)
            if NO_AVG:
                loss = t2t_loss
            else:
                loss = t2t_loss / avg_textlen
            self.gradient_decent(loss, self.opts['t2t'], freeze_att=FREEZE_ATT)

            loss_data = loss.item()
            logging.info("loss: %f", loss_data)
            chainer.reporter.report({'t/loss': loss_data})
            chainer.reporter.report({'t/t2t_loss': loss_data})
            chainer.reporter.report({'t/t2t_acc': t2t_acc})
        else:
            logging.error("Error: cannot find correct mode ('p', 'a', 't')")
            sys.exit()

        delete_feat(data)


class DataParallel(torch.nn.DataParallel):
    def scatter(self, inputs, kwargs, device_ids, dim):
        r"""Scatter with support for kwargs dictionary"""
        if len(inputs) == 1:
            inputs = inputs[0]
        avg = int(math.ceil(len(inputs) * 1. / len(device_ids)))
        # inputs = scatter(inputs, device_ids, dim) if inputs else []
        inputs = [[inputs[i:i + avg]] for i in range(0, len(inputs), avg)]
        kwargs = torch.nn.scatter(kwargs, device_ids, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids, self.dim)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)


def extract_json(json, mode):
    all_utt = {}
    for utt in json:
        if json[utt]['utt2mode'] == mode:
            all_utt[utt]= json[utt]

    return all_utt


def make_batchset_asrtts(data, batch_size, max_length_in, max_length_out, num_batches=0, factor_audio=1, factor_text=4):

    data_audio_only = extract_json(data, 'a')
    data_text_only = extract_json(data, 't')
    data_parallel = extract_json(data, 'p')

    if len(data_audio_only) > 0:
        json_audio_only = make_batchset(data_audio_only, factor_audio * batch_size,
                                        max_length_in, max_length_out, num_batches)
    else:
        json_audio_only = []
    if len(data_text_only) > 0:
        json_text_only = make_batchset(data_text_only, factor_text * batch_size,
                                       max_length_in, max_length_out, num_batches)
    else:
        json_text_only = []
    if len(data_parallel) > 0:
        json_parallel = make_batchset(data_parallel, batch_size, max_length_in,
                                      max_length_out, num_batches)
    else:
        json_parallel = []

    return json_audio_only + json_text_only + json_parallel


def setup_tts_loss(odim_asr, idim_tts, args):
    from argparse import Namespace
    # define output activation function
    if args.tts_output_activation is None:
        output_activation_fn = None
    elif hasattr(torch.nn.functional, args.tts_output_activation):
        output_activation_fn = getattr(torch.nn.functional, args.tts_output_activation)
    else:
        raise ValueError('there is no such an activation function. (%s)' % args.tts_output_activation)

    tts_args = Namespace(
        spk_embed_dim=args.tts_spk_embed_dim,
        embed_dim=args.tts_embed_dim,
        elayers=args.tts_elayers,
        eunits=args.tts_eunits,
        econv_layers=args.tts_econv_layers,
        econv_chans=args.tts_econv_chans,
        econv_filts=args.tts_econv_filts,
        dlayers=args.tts_dlayers,
        dunits=args.tts_dunits,
        prenet_layers=args.tts_prenet_layers,
        prenet_units=args.tts_prenet_units,
        postnet_layers=args.tts_postnet_layers,
        postnet_chans=args.tts_postnet_chans,
        postnet_filts=args.tts_postnet_filts,
        output_activation=output_activation_fn,
        adim=args.tts_adim,
        aconv_chans=args.tts_aconv_chans,
        aconv_filts=args.tts_aconv_filts,
        cumulate_att_w=args.tts_cumulate_att_w,
        use_batch_norm=args.tts_use_batch_norm,
        use_concate=args.tts_use_concate,
        dropout=args.tts_dropout_rate,
        zoneout=args.tts_zoneout_rate,
        monotonic=args.tts_monotonic)
    e2e_tts = Tacotron2(
        idim=odim_asr,
        odim=idim_tts,
        args=tts_args)
    return Tacotron2Loss(
        model=e2e_tts,
        use_masking=args.tts_use_masking,
        bce_pos_weight=args.tts_bce_pos_weight)

def train(args):
    '''Run training'''
    # seed setting
    torch.manual_seed(args.seed)

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # revmoe type check
    if args.debugmode < 2:
        chainer.config.type_check = False
        logging.info('torch type check is disabled')
    # use determinisitic computation or not
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = False
        logging.info('torch cudnn deterministic is disabled')
    else:
        torch.backends.cudnn.deterministic = True

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim_asr = int(valid_json[utts[0]]['input'][0]['shape'][1])
    idim_tts = int(valid_json[utts[0]]['input'][1]['shape'][1])
    odim_asr = int(valid_json[utts[0]]['output'][0]['shape'][1])
    assert idim_tts == idim_asr - 3
    logging.info('#input dims for ASR: ' + str(idim_asr))
    logging.info('#input dims for TTS: ' + str(idim_tts))
    logging.info('#output dims: ' + str(odim_asr))
    if args.tts_use_speaker_embedding:
        args.tts_spk_embed_dim = int(valid_json[utts[0]]['input'][2]['shape'][0])
    else:
        args.tts_spk_embed_dim = None

    # specify attention, CTC, hybrid mode
    if args.mtlalpha == 1.0:
        mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    # specify model architecture for ASR
    e2e_asr = E2E(idim_asr, odim_asr, args)
    logging.info(e2e_asr)
    asr_loss = Loss(e2e_asr, args.mtlalpha)

    # specify model architecture for TTS
    # reverse input and output dimension
    tts_loss = setup_tts_loss(odim_asr, idim_tts, args)
    logging.info(tts_loss)

    # define loss
    model = ASRTTSLoss(asr_loss, tts_loss, args)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.conf'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        # TODO(watanabe) use others than pickle, possibly json, and save as a text
        pickle.dump((idim_asr, odim_asr, args), f)
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # Set gpu
    ngpu = args.ngpu
    if ngpu == 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
    elif ngpu > 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model = torch.nn.DataParallel(model, device_ids=gpu_id)
        model.cuda()
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu
    else:
        gpu_id = [-1]

    # Setup an optimizer
    dummy_target = chainer.Chain()
    opts = {}
    #opts['asr'] = torch.optim.Adadelta(model.asr_loss.parameters(), rho=0.95, eps=args.eps)
    if ngpu > 1:
        module = model.module
    else:
        module = model

    optim_class = getattr(torch.optim, args.optim)
    opts['asr'] = optim_class(module.asr_loss.parameters(), args.lr*args.asr_weight, eps=args.eps, weight_decay=args.weight_decay)
    opts['tts'] = optim_class(module.tts_loss.parameters(), args.lr*args.tts_weight, eps=args.eps, weight_decay=args.weight_decay)
    opts['s2s'] = optim_class(module.ae_speech.parameters(), args.lr*args.s2s_weight, eps=args.eps, weight_decay=args.weight_decay)
    opts['t2t'] = optim_class(module.ae_text.parameters(), args.lr*args.t2t_weight, eps=args.eps, weight_decay=args.weight_decay)
    ae_param = list(set(list(module.ae_speech.parameters()) + list(module.ae_text.parameters())))
    opts['mmd'] = optim_class(ae_param, args.lr*args.mmd_weight, eps=args.eps, weight_decay=args.weight_decay)
    for key in ['asr', 'tts', 's2s', 't2t', 'mmd']:
        # FIXME: TOO DIRTY HACK
        setattr(opts[key], "target", dummy_target)
        setattr(opts[key], "serialize", lambda s: dummy_target.serialize(s))

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # read utt2mode scp
    def load_utt2mode(scp, utts):
        with open(scp, 'r') as f:
            for line in f:
                k, v = line.strip().split()
                if k in utts.keys():
                    utts[k]['utt2mode'] = v
    load_utt2mode(args.train_utt2mode, train_json)
    load_utt2mode(args.valid_utt2mode, valid_json)


    # make minibatch list (variable length)
    train = make_batchset_asrtts(train_json, args.batch_size,
                                 args.maxlen_in, args.maxlen_out, args.minibatches)
    valid = make_batchset_asrtts(valid_json, args.batch_size,
                                 args.maxlen_in, args.maxlen_out, args.minibatches)
    # hack to make batchsze argument as 1
    # actual bathsize is included in a list
    train_iter = chainer.iterators.SerialIterator(train, 1)
    valid_iter = chainer.iterators.SerialIterator(
        valid, 1, repeat=False, shuffle=False)

    # Set up a trainer
    updater = CustomUpdater(model, args.grad_clip, train_iter, opts,
                            converter=converter_kaldi, device=gpu_id, args=args)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
        if ngpu > 1:
            model.module.load_state_dict(torch.load(args.outdir + '/model.acc.best'))
        else:
            model.load_state_dict(torch.load(args.outdir + '/model.acc.best'))
    if args.model:
        if ngpu > 1:
            model.module.load_state_dict(torch.load(args.model))
        else:
            model.load_state_dict(torch.load(args.model))
    elif args.model_asr:
        if ngpu > 1:
            model.asr_loss.module.load_state_dict(torch.load(args.model_asr))
        else:
            model.asr_loss.load_state_dict(torch.load(args.model_asr))
    elif args.model_tts:
        if ngpu > 1:
            model.tts_loss.module.load_state_dict(torch.load(args.model_tts))
        else:
            model.tts_loss.load_state_dict(torch.load(args.model_tts))
        model = trainer.updater.model

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluater(
        model, valid_iter, dummy_target, converter=converter_kaldi, device=gpu_id))

    # Save attention weight each epoch
    if args.num_save_attention > 0 and args.mtlalpha != 1.0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        data = converter_kaldi([data], device=gpu_id, use_speaker_embedding=args.tts_spk_embed_dim)
        trainer.extend(CustomPlotAttentionReport(
            model, True,
            data, args.outdir + "/att_ws_asr"), trigger=(1, 'epoch'))
        trainer.extend(CustomPlotAttentionReport(
            model, False,
            data, args.outdir + '/att_ws_tts', reverse=True), trigger=(1, 'epoch'))

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    # report keys
    report_keys = ['t/loss', 't/tts_loss', 't/s2s_loss', 't/asr_loss', 't/t2t_loss', 't/mmd_loss',
                   'd/loss', 'd/tts_loss', 'd/s2s_loss', 'd/asr_loss', 'd/t2t_loss']
    trainer.extend(extensions.PlotReport(report_keys,
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['t/asr_acc', 'd/asr_acc'],
                                         'epoch', file_name='acc.png'))

    # Save best models
    def torch_save(path, _):
        if ngpu > 1:
            torch.save(model.module.state_dict(), path)
            torch.save(model.module, path + ".pkl")
        else:
            torch.save(model.state_dict(), path)
            torch.save(model, path + ".pkl")

    trainer.extend(extensions.snapshot_object(model, 'model.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('d/loss'))

    def torch_load(path, obj):
        if ngpu > 1:
            model.module.load_state_dict(torch.load(path))
        else:
            model.load_state_dict(torch.load(path))
        return obj

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(REPORT_INTERVAL, 'iteration')))
    report_keys = ['t/loss', 't/tts_loss', 't/s2s_loss', 't/asr_loss', 't/t2t_loss', 't/asr_acc', 't/t2t_acc', 't/mmd_loss',
                   'd/loss', 'd/tts_loss', 'd/s2s_loss', 'd/asr_loss', 'd/t2t_loss', 'd/asr_acc', 'd/t2t_acc']
    report_keys.append('epoch')
    report_keys.append('iteration')
    report_keys.append('elapsed_time')
    trainer.extend(extensions.PrintReport(report_keys), trigger=(REPORT_INTERVAL, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))

    # Run the training
    trainer.run()


def recog(args):
    '''Run recognition'''

    # rnnlm
    import extlm_pytorch
    import lm_pytorch

    # seed setting
    torch.manual_seed(args.seed)

    # read training config
    with open(args.model_conf, "rb") as f:
        logging.info('reading a model config file from' + args.model_conf)
        idim_asr, odim_asr, train_args = pickle.load(f)

    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # specify model architecture
    logging.info('reading model parameters from' + args.model)
    e2e_asr = E2E(idim_asr, odim_asr, train_args)
    logging.info(e2e_asr)
    asr_loss = Loss(e2e_asr, train_args.mtlalpha)

    # specify model architecture for TTS
    # reverse input and output dimension
    tts_loss = setup_tts_loss(odim_asr, idim_asr - 3, train_args)
    logging.info(tts_loss)

    # define loss
    model = ASRTTSLoss(asr_loss, tts_loss, train_args)

    def cpu_loader(storage, location):
        return storage

    def remove_dataparallel(state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        return new_state_dict

    model.load_state_dict(remove_dataparallel(torch.load(args.model, map_location=cpu_loader)))

    # read rnnlm
    if args.rnnlm:
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(len(train_args.char_list), 650))
        rnnlm.load_state_dict(torch.load(args.rnnlm, map_location=cpu_loader))
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        if not args.word_dict:
            logging.error('word dictionary file is not specified for the word RNNLM.')
            sys.exit(1)

        word_dict = load_labeldict(args.word_dict)
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(len(word_dict), 650))
        word_rnnlm.load_state_dict(torch.load(args.word_rnnlm, map_location=cpu_loader))
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(word_rnnlm.predictor,
                                           rnnlm.predictor, word_dict, char_dict))
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor,
                                              word_dict, char_dict))

    # read json data
    with open(args.recog_json, 'rb') as f:
        recog_json = json.load(f)['utts']

    new_json = {}
    for name in recog_json.keys():
        feat = kaldi_io_py.read_mat(recog_json[name]['input'][0]['feat'])
        nbest_hyps = e2e_asr.recognize(feat, args, train_args.char_list, rnnlm=rnnlm)
        # get 1best and remove sos
        y_hat = nbest_hyps[0]['yseq'][1:]
        y_true = map(int, recog_json[name]['output'][0]['tokenid'].split())

        # print out decoding result
        seq_hat = [train_args.char_list[int(idx)] for idx in y_hat]
        seq_true = [train_args.char_list[int(idx)] for idx in y_true]
        seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
        seq_true_text = "".join(seq_true).replace('<space>', ' ')
        logging.info("groundtruth[%s]: " + seq_true_text, name)
        logging.info("prediction [%s]: " + seq_hat_text, name)

        # copy old json info
        new_json[name] = dict()
        new_json[name]['utt2spk'] = recog_json[name]['utt2spk']

        # added recognition results to json
        logging.debug("dump token id")
        out_dic = dict()
        for _key in recog_json[name]['output'][0]:
            out_dic[_key] = recog_json[name]['output'][0][_key]

        # TODO(karita) make consistent to chainer as idx[0] not idx
        out_dic['rec_tokenid'] = " ".join([str(idx) for idx in y_hat])
        logging.debug("dump token")
        out_dic['rec_token'] = " ".join(seq_hat)
        logging.debug("dump text")
        out_dic['rec_text'] = seq_hat_text

        new_json[name]['output'] = [out_dic]
        # TODO(nelson): Modify this part when saving more than 1 hyp is enabled
        # add n-best recognition results with scores
        if args.beam_size > 1 and len(nbest_hyps) > 1:
            for i, hyp in enumerate(nbest_hyps):
                y_hat = hyp['yseq'][1:]
                seq_hat = [train_args.char_list[int(idx)] for idx in y_hat]
                seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
                new_json[name]['rec_tokenid' + '[' + '{:05d}'.format(i) + ']'] = " ".join([str(idx) for idx in y_hat])
                new_json[name]['rec_token' + '[' + '{:05d}'.format(i) + ']'] = " ".join(seq_hat)
                new_json[name]['rec_text' + '[' + '{:05d}'.format(i) + ']'] = seq_hat_text
                new_json[name]['score' + '[' + '{:05d}'.format(i) + ']'] = hyp['score']

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_json}, indent=4, sort_keys=True).encode('utf_8'))
