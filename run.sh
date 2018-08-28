#!/bin/bash

. ./path.sh
. ./cmd.sh

sup_data_ratio=1.0

# general configuration
init=""
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
gpu=-1         # use 0 when using GPU on slurm/grid engine, otherwise -1
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option

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

use_batchnorm=False

# loss
unsupervised_loss=mmd
sup_loss_ratio=0.5
st_ratio=0.5

# speech/text unsupervised loss ratio
mtlalpha=0.5
paired_hidden=False

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15
lr=1.0
weight_decay=0.0

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

# exp tag
tag="" # tag for managing experiments.

train_set=train_si84
unpaired_set=train_si284

. utils/parse_options.sh || exit 1;

. ./path.sh 
. ./cmd.sh 

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_dev=test_dev93
recog_set="test_dev93 test_eval92"

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    ./shell/wsj_format_data_with_si84.sh
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_us_dir=${dumpdir}/${unpaired_set}/delta${do_delta}; mkdir -p ${feat_us_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${train_set} ${unpaired_set} ${recog_set} ; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${unpaired_set}/feats.scp data/${unpaired_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${unpaired_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${unpaired_set}/feats.scp data/${unpaired_set}/cmvn.ark exp/dump_feats/train ${feat_us_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${unpaired_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
fi

dict=data/lang_1char/${unpaired_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${unpaired_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${unpaired_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_us_dir}/feats.scp --nlsyms ${nlsyms} \
                 data/${unpaired_set} ${dict} > ${feat_us_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
fi

if [ -z ${tag} ]; then
    expdir=exp/semi_data${sup_data_ratio}_${unsupervised_loss}_loss${sup_loss_ratio}_${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_lr${lr}_wd_${weight_decay}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_epochs${epochs}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${tag}
fi
mkdir -p ${expdir}

init_train_script=asr_train_loop_th.py
retrain_script=retrain_loop_th.py
decode_script=unsupervised_recog_th.py

if [ ${stage} -le 3 ]; then
    echo "stage 3: Network Init-Training"

    ${cuda_cmd} ${expdir}/init_train.log \
        ${init_train_script} \
        --gpu ${gpu} \
        --outdir ${expdir}/init_results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --train-feat scp:${feat_tr_dir}/feats.scp \
        --valid-feat scp:${feat_dt_dir}/feats.scp \
        --train-label ${feat_tr_dir}/data.json \
        --valid-label ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --supervised-data-ratio ${sup_data_ratio} \
        --epochs ${epochs}
fi

if [ -z $init ]; then
    init=${expdir}/init_results/model.${recog_model}
fi


if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Re-Training"

    ${cuda_cmd} ${expdir}/train.log \
                ${retrain_script} \
                --init-model ${init} \
        --gpu ${gpu} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --train-feat scp:${feat_tr_dir}/feats.scp \
        --valid-feat scp:${feat_dt_dir}/feats.scp \
        --unsupervised-feat scp:${feat_us_dir}/feats.scp \
        --train-label ${feat_tr_dir}/data.json \
        --valid-label ${feat_dt_dir}/data.json \
        --unsupervised-json ${feat_us_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --lr ${lr} \
        --weight-decay ${weight_decay} \
        --unsupervised-loss ${unsupervised_loss} \
        --supervised-loss-ratio ${sup_loss_ratio} \
        --supervised-data-ratio ${sup_data_ratio} \
        --speech-text-ratio ${st_ratio} \
        --use-batchnorm ${use_batchnorm} \
        --epochs ${epochs}
fi


if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding retrained model"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}

        # split data
        data=data/${rtask}
        sdata=${data}/split${nj}utt;
        if [ ! -d $sdata ]; then
            split_data.sh --per-utt ${data} ${nj};
        fi

        # feature extraction
        feats="ark,s,cs:apply-cmvn --norm-vars=true data/train_si284/cmvn.ark scp:${sdata}/JOB/feats.scp ark:- |"
        if ${do_delta}; then
            feats="$feats add-deltas ark:- ark:- |"
        fi

        if [ ! -e ${data}/data.json ]; then
            # make json labels for recognition
            data2json.sh --nlsyms ${nlsyms} ${data} ${dict} > ${data}/data.json
        fi

        #### use CPU for decoding
        gpu=-1

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            ${decode_script} \
            --gpu ${gpu} \
            --recog-feat "$feats" \
            --recog-label ${data}/data.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} &
        wait

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
    ) &
    done
    wait
    echo "Finished"
fi


if [ ${stage} -le 6 ]; then
    echo "stage 6: Decoding init model"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=init_decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}

        # split data
        data=data/${rtask}
        sdata=${data}/split${nj}utt;
        if [ ! -d $sdata ]; then
            split_data.sh --per-utt ${data} ${nj};
        fi

        # feature extraction
        feats="ark,s,cs:apply-cmvn --norm-vars=true data/train_si284/cmvn.ark scp:${sdata}/JOB/feats.scp ark:- |"
        if ${do_delta}; then
            feats="$feats add-deltas ark:- ark:- |"
        fi

        if [ ! -e ${data}/data.json ]; then
            # make json labels for recognition
            data2json.sh --nlsyms ${nlsyms} ${data} ${dict} > ${data}/data.json
        fi

        #### use CPU for decoding
        gpu=-1

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog_th.py \
            --gpu ${gpu} \
            --recog-feat "$feats" \
            --recog-label ${data}/data.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/init_results/model.${recog_model}  \
            --model-conf ${expdir}/init_results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} &
        wait

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
    ) &
    done
    wait
    echo "Finished"
fi

