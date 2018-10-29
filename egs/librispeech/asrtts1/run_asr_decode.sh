#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
gpu=           # will be deprecated, please use ngpu
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=64
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
seed=1         # random seed number
resume=        # Resume the training from snapshot
exp_prefix=""
# feature configuration
do_delta=false # true when using CNN
fs=16000       # sampling frequency
fmax=""        # maximum frequency
fmin=""        # minimum frequency
n_mels=80      # number of mel basis
n_fft=1024     # number of fft points
n_shift=512    # number of shift points
win_length=""  # number of samples in analysis window

# ASR network archtecture
# encoder related
etype=vggblstmp   # encoder architecture type
elayers=4
eunits=320
eprojs=512 # ASR default 320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
adim=320
atype=location
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.0

# TTS network archtecture
# encoder related
tts_embed_dim=512
tts_elayers=1
tts_eunits=512
tts_econv_layers=3 # if set 0, no conv layer is used
tts_econv_chans=512
tts_econv_filts=5
# decoder related
tts_dlayers=2
tts_dunits=1024
tts_prenet_layers=2  # if set 0, no prenet is used
tts_prenet_units=256
tts_postnet_layers=5 # if set 0, no postnet is used
tts_postnet_chans=512
tts_postnet_filts=5
tts_use_speaker_embedding=true
# attention related
tts_adim=128
tts_aconv_chans=32
tts_aconv_filts=15      # resulting in filter_size = aconv_filts * 2 + 1
tts_cumulate_att_w=true # whether to cumulate attetion weight
tts_use_batch_norm=true # whether to use batch normalization in conv layer
tts_use_concate=true    # whether to concatenate encoder embedding with decoder lstm outputs
tts_use_residual=false  # whether to use residual connection in encoder convolution
tts_use_masking=true    # whether to mask the padded part in loss calculation
tts_bce_pos_weight=1.0  # weight for positive samples of stop token in cross-entropy calculation

tts_dropout=0.5
tts_zoneout=0.1

# common configurations

# minibatch related
batchsize=32
batch_sort_key="" # empty or input or output (if empty, shuffled batch will be used)
maxlen_in=400  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=Adam
lr=1e-3
eps=1e-6
weight_decay=0.0
epochs=10
asr_weight=0.1
tts_weight=1.0
s2s_weight=0.01
t2t_weight=0.01
mmd_weight=0.01
inter_domain_loss=mmd
use_mmd_ae=True
# rnnlm related
lm_weight=0.3

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.8
minlenratio=0.3
ctc_weight=0.0
recog_model=loss.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data
datadir=../asr1/data
datadir=/data/rigel1/corpora/LibriSpeech
# model
model_asr=
model_tts=
model=

# base url for downloads.
data_url=www.openslr.org/resources/12

# exp tag
tag="" # tag for managing experiments.
expdir=""

data_type=data_short_3000 # data or data_short or data_short_p

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# check gpu option usage
if [ ! -z ${gpu} ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ ${gpu} -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
train_dev=dev_clean
recog_set="test_clean dev_clean" # test_other dev_clean dev_other"
dict=data/lang_1char/train_clean_100_units.txt
lmexpdir=exp/train_rnnlm_2layer_bs256


if [ ${stage} -le 6 ]; then
    echo "stage 6: Decoding"

    for rtask in ${recog_set}; do
    (
	rtask=${rtask}_asr
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        if [ ! -e ${feat_recog_dir}/split${nj}utt/data.1.json ]; then
            splitjson.py --parts ${nj} ${feat_recog_dir}/data.json
        fi

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asrtts_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --lm-weight ${lm_weight}

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}

    )
    done
    wait
    echo "Finished"
fi

