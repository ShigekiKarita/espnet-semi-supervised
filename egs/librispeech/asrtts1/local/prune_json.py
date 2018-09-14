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
        if j['utts'][utt_id]['utt2mode'] == 'p': 
            new_dic[utt_id] = j['utts'][utt_id]
    jsonstring = json.dumps({'utts': new_dic},
                            indent=4,
                            ensure_ascii=False,
                            sort_keys=True).encode('utf_8')
    print(jsonstring)
