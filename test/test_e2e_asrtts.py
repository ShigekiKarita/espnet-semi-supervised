#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import pytest
import torch

from argparse import Namespace

from e2e_asr_th import E2E
from e2e_asr_th import Loss
from e2e_asr_th import pad_list

from e2e_tts_th import Tacotron2
from e2e_tts_th import Tacotron2Loss

from e2e_asrtts_th import packed_mmd
from e2e_asrtts_th import ASRTTSLoss
from e2e_asrtts_th import get_tts_data

from test_e2e_asr import prepare_inputs as prepare_asr_inputs
from test_e2e_tts import prepare_inputs as prepare_tts_inputs


def make_args(**kwargs):
    from test_e2e_asr import make_arg
    asr_defaults = vars(make_arg())

    # setting from egs/librispeech/asrtts1/run.sh
    asr_tts_defaults = dict(
        etype="blstmp",
        elayers=4,
        eunits=320,
        eprojs=512,
        subsample="1_2_2_1_1",
        dlayers=1,
        dunits=300,
        atype="location",
        adim=320,
        aconv_chans=10,
        aconv_filts=100,
        mtlalpha=0.0,
        tts_embed_dim=512,
        tts_elayers=1,
        tts_eunits=512,
        tts_econv_layers=3,
        tts_econv_chans=512,
        tts_econv_filts=5,
        tts_dlayers=2,
        tts_dunits=1024,
        tts_prenet_layers=2,
        tts_prenet_units=256,
        tts_postnet_layers=5,
        tts_postnet_chans=512,
        tts_postnet_filts=5,
        tts_adim=128,
        tts_aconv_chans=32,
        tts_aconv_filts=15,
        tts_cumulate_att_w=True,
        tts_use_speaker_embedding=True,
        tts_use_batch_norm=True,
        tts_use_concate=True,
        tts_use_residual=False,
        tts_use_masking=True,
        tts_bce_pos_weight=1.0,
        tts_dropout_rate=0.5,
        tts_zoneout_rate=0.1,
        tts_monotonic=0.0,
        tts_output_activation=None,
        lr=1e-3,
        eps=1e-6,
        weight_decay=0.0,
        batch_sort_key=None,
        batch_size=64,
        maxlen_in=400,
        maxlen_out=150,
        opt="adam",
        epochs=30,
        mmd_weight=1.0
    )
    defaults = dict()
    defaults.update(asr_defaults)
    defaults.update(asr_tts_defaults)
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_asrtts_model_trainable_and_decodable():
    from asrtts_pytorch import setup_tts_loss
    from test_e2e_tts import make_inference_args
    args = make_args()
    print(args)

    # asr setup
    asr_xpad, asr_xlens, asr_ypad = prepare_asr_inputs("pytorch")
    asr_batchsize = asr_xpad.shape[0]
    asr_idim = 40
    asr_odim = 5
    asr_model = Loss(E2E(asr_idim, asr_odim, args), args.mtlalpha)
    # asr trainable
    asr_loss = asr_model(asr_xpad, asr_xlens, asr_ypad)
    asr_loss.backward()
    # asr decodable
    asr_model.eval()
    with torch.no_grad():
        in_data = np.random.randn(100, 40)
        asr_model.predictor.recognize(in_data, args, args.char_list)  # decodable
    asr_model.train()

    # tts setup
    tts_batchsize = asr_batchsize # 2
    tts_maxin_len = 10
    tts_maxout_len = 10
    tts_batch = prepare_tts_inputs(tts_batchsize, asr_odim, asr_idim-3, tts_maxin_len, tts_maxout_len)
    tts_xpad, tts_ilens, tts_ypad, tts_labels, tts_olens = tts_batch
    setattr(args, "tts_spk_embed_dim", 2)
    spembs = torch.randn(tts_batchsize, args.tts_spk_embed_dim)
    tts_model = setup_tts_loss(asr_odim, asr_idim - 3, args)
    # tts trainable
    tts_loss = tts_model(*tts_batch, spembs)
    tts_loss.backward()  # trainable
    # tts decodable
    tts_model.eval()
    with torch.no_grad():
        spemb = spembs[0]
        x = tts_xpad[0][:tts_ilens[0]]
        yhat, probs, att_ws = tts_model.model.inference(x, Namespace(**make_inference_args()), spemb)
        att_ws = tts_model.model.calculate_all_attentions(tts_xpad, tts_ilens, tts_ypad, spembs)
    tts_model.train()

    # asrtts model trainable
    model = ASRTTSLoss(asr_model, tts_model, args)
    opts = {}
    #opts['asr'] = torch.optim.Adadelta(model.asr_loss.parameters(), rho=0.95, eps=args.eps)
    opts['asr'] = torch.optim.Adam(model.asr_loss.parameters(), args.lr*0.1, eps=args.eps, weight_decay=args.weight_decay)
    opts['tts'] = torch.optim.Adam(model.tts_loss.parameters(), args.lr, eps=args.eps, weight_decay=args.weight_decay)
    opts['s2s'] = torch.optim.Adam(model.ae_speech.parameters(), args.lr*0.01, eps=args.eps, weight_decay=args.weight_decay)
    opts['t2t'] = torch.optim.Adam(model.ae_text.parameters(), args.lr*0.01, eps=args.eps, weight_decay=args.weight_decay)

    ae_param = list(set(list(model.ae_speech.parameters()) + list(model.ae_text.parameters())))
    opts['mmd'] = torch.optim.Adam(ae_param, args.lr*0.01, eps=args.eps, weight_decay=args.weight_decay)

    # data prep
    dummy_tokenid = "1 2 3"
    asr_data = [(
        "tmp",
        dict(
            feat_asr=asr_xpad[i, :asr_xlens[i]].numpy(),
            feat_tts=tts_ypad[i, :tts_olens[i]].numpy(),
            feat_spembs=spembs[0].numpy(),
            output=[
                dict(
                    tokenid=dummy_tokenid, # " ".join(map(str, asr_ypad[i].tolist())),
                    shape=[3, asr_odim]
                )
            ]
        )  # dict
    ) # tuple
    for i in range(asr_batchsize)]
    tts_data = []

    # speech-to-speech
    s2s_loss, hspad, hslen = model.ae_speech(asr_data, return_hidden=True)
    s2s_loss.backward(retain_graph=True)
    # text-to-text
    t2t_loss, t2t_acc, htpad, htlen = model.ae_text(asr_data, return_hidden=True)
    t2t_loss.backward(retain_graph=True)
    # inter-domain loss
    mmd_loss = packed_mmd(hspad, hslen, htpad, htlen)
    mmd_loss.backward()
