#!/usr/bin/env bash

dumpdir=dump
do_delta=false
task=dev_clean
nnet_dir=exp/xvector_nnet_1a
feat_dir=${dumpdir}/${task}/delta${do_delta}; mkdir -p ${feat_dir}

local/multi_jsons.py ${dumpdir}/${task}_asr/delta${do_delta}/data.json ${dumpdir}/${task}_tts/delta${do_delta}/data.json \
			         > ${feat_dir}/data.json

local/update_json.sh ${dumpdir}/${task}/delta${do_delta}/data.json ${nnet_dir}/xvectors_${task}/xvector.scp

python2 local/remove_longshort_utt.py \
	    --max-input 3000 --max-output 400 \
	    ${feat_dir}/data.json > ${feat_dir}/data_short_3000.json
