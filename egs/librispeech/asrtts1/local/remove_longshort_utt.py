#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function

import argparse
import json
import logging
import os
import sys

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str,
                        help='json file')
    parser.add_argument('--max-input', type=int, default=3000,
                        help='remove utterance less than max_input. default=3000')
    parser.add_argument('--max-output', type=int, default=400,
                        help='remove utterance less than max_output. default=400')
    parser.add_argument('--min-input', type=int, default=0,
                        help='remove utterance less than min_input. default=0')
    parser.add_argument('--min-output', type=int, default=0,
                        help='remove utterance less than min_output. default=0')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # load json and split keys
    j = json.load(open(args.json))
    utt_ids = j['utts'].keys()
    logging.info("number of utterances = %d" % len(utt_ids))
    new_dic = dict()
    for utt_id in utt_ids:
        if j['utts'][utt_id]['input'][0]['shape'][0] < args.max_input \
           and j['utts'][utt_id]['input'][0]['shape'][0] > args.min_input \
           and j['utts'][utt_id]['output'][0]['shape'][0] < args.max_output \
           and j['utts'][utt_id]['output'][0]['shape'][0] > args.min_output:
            new_dic[utt_id] = j['utts'][utt_id]
    jsonstring = json.dumps({'utts': new_dic},
                            indent=4,
                            ensure_ascii=False,
                            sort_keys=True).encode('utf_8')
    print(jsonstring)
