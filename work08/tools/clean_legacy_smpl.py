# -*- coding: utf-8 -*-
"""One-time cleaner for legacy SMPL .pkl files.
Run this script only in a Python 2.7 environment with numpy, scipy and chumpy.
It replaces Chumpy objects in a legacy model pickle with plain NumPy arrays.
"""
from __future__ import print_function

import argparse
import os
import os.path as osp
import pickle
import numpy as np


def is_chumpy_object(value):
    return 'chumpy' in str(type(value)).lower()


def main():
    parser = argparse.ArgumentParser(
        description='Remove legacy Chumpy objects from a SMPL pickle file.'
    )
    parser.add_argument('--input', required=True, help='Path to legacy SMPL_NEUTRAL.pkl')
    parser.add_argument('--output', required=True, help='Path for cleaned SMPL_NEUTRAL.pkl')
    args = parser.parse_args()

    if not osp.isfile(args.input):
        raise IOError('Input file not found: {0}'.format(args.input))

    out_dir = osp.dirname(osp.abspath(args.output))
    if out_dir and not osp.isdir(out_dir):
        os.makedirs(out_dir)

    print('Reading: {0}'.format(args.input))
    with open(args.input, 'rb') as handle:
        data = pickle.load(handle)

    cleaned = {}
    converted = []
    for key, value in data.iteritems():
        if is_chumpy_object(value):
            cleaned[key] = np.array(value)
            converted.append(key)
        else:
            cleaned[key] = value

    with open(args.output, 'wb') as handle:
        pickle.dump(cleaned, handle, protocol=2)

    print('Wrote cleaned model: {0}'.format(args.output))
    if converted:
        print('Converted Chumpy keys: {0}'.format(', '.join(converted)))
    else:
        print('No top-level Chumpy key was detected, but the file was rewritten successfully.')


if __name__ == '__main__':
    main()
