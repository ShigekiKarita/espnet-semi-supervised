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
    parser.add_argument('jsons', type=str, nargs='+', help='json files')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    js = []
    intersec_ks = []
    for x in args.jsons:
        with open(x, 'r') as f:
            j = json.load(f)
        ks = j['utts'].keys()
        logging.info(x + ': has ' + str(len(ks)) + ' utterances')
        if len(intersec_ks) > 0:
            intersec_ks = intersec_ks.intersection(set(ks))
        else:
            intersec_ks = set(ks)
        js.append(j)
    logging.info('new json has ' + str(len(intersec_ks)) + ' utterances')

    new_dic = dict()
    for utt_id in intersec_ks:
        new_dic[utt_id] = js[0]['utts'][utt_id]
        for i in range(1, len(args.jsons)):
            new_dic[utt_id]['input'].append(js[i]['utts'][utt_id]['input'][0])
            new_dic[utt_id]['output'].append(js[i]['utts'][utt_id]['output'][0])
            new_dic[utt_id]['input'][i]['name'] = 'input' + str(i + 1)
            new_dic[utt_id]['output'][i]['name'] = 'target' + str(i + 1)
    jsonstring = json.dumps({'utts': new_dic},
                            indent=4,
                            ensure_ascii=False,
                            sort_keys=True).encode('utf_8')
    print(jsonstring)
