#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

expdir=""

# general configuration
init=""
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
gpu=-1         # use 0 when using GPU on slurm/grid engine, otherwise -1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=6
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aconv_chans=10
aconv_filts=100


# loss
unsupervised_loss=None
sup_loss_ratio=0.5
# speech/text unsupervised loss ratio
mtlalpha=0.5

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15
lr=1.0
weight_decay=0.0

# rnnlm related
lm_weight=0.1

# decoding parameter
beam_size=20
penalty=0.1
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# data
# wsj0=/export/corpora5/LDC/LDC93S6B
# wsj1=/export/corpora5/LDC/LDC94S13B
wsj0=/nfs/kswork/kishin/karita/datasets/LDC93S6A
wsj1=/nfs/kswork/kishin/karita/datasets/LDC94S13A

# wsj0=/data/rigel1/corpora/LDC93S6A
# wsj1=/data/rigel1/corpora/LDC94S13A

lmexpdir="None"
# exp tag
tag="" # tag for managing experiments.

train_set=train_si84
unpaired_set=train_si284
decode_script=unsupervised_recog_th.py

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh 

dict=data/lang_1char/train_si284_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt
recog_set="test_dev93 test_eval92"


if [ $lmexpdir = "None" ]; then
    rnnlmopt=""
else
    rnnlmopt="--rnnlm ${lmexpdir}/rnnlm.model.best --lm-weight ${lm_weight} "
fi

echo "stage 5: Decoding"
nj=32

rnnexpdir=${expdir}/rnnlm${lm_weight}
mkdir -p $rnnexpdir
for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}

        # split data
        data=data/${rtask}
        # split_data.sh --per-utt ${data} ${nj};
        sdata=${data}/split${nj}utt;

        # feature extraction
        feats="ark,s,cs:apply-cmvn --norm-vars=true data/train_si284/cmvn.ark scp:${sdata}/JOB/feats.scp ark:- |"
        if ${do_delta}; then
            feats="$feats add-deltas ark:- ark:- |"
        fi

        # make json labels for recognition
        # data2json.sh --nlsyms ${nlsyms} ${data} ${dict} > ${data}/data.json

        #### use CPU for decoding
        gpu=-1

        ${decode_cmd} JOB=1:${nj} ${rnnexpdir}/${decode_dir}/log/decode.JOB.log \
                      ${decode_script} \
                      ${rnnlmopt} --gpu ${gpu} \
                      --recog-feat "$feats" \
                      --recog-label ${data}/data.json \
                      --result-label ${rnnexpdir}/${decode_dir}/data.JOB.json \
                      --model ${expdir}/results/model.${recog_model}  \
                      --model-conf ${expdir}/results/model.conf  \
                      --beam-size ${beam_size} \
                      --penalty ${penalty} \
                      --maxlenratio ${maxlenratio} \
                      --minlenratio ${minlenratio} \
                      --verbose ${verbose} \
                      --ctc-weight ${ctc_weight} &
        wait

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${rnnexpdir}/${decode_dir} ${dict}
        
    ) &
done
wait
echo "Finished"
