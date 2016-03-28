from __future__ import print_function
import argparse
import os
import numpy as np
import sys
from matplotlib import pyplot as plt

# run system commands
from subprocess import call


def easy_call(command):
    try:
        call(command, shell=True)
    except Exception as exception:
        print("Error: could not execute the following")
        print(">>", command)
        print(type(exception))  # the exception instance
        print(exception.args)  # arguments stored in .args
        exit(-1)


def read_data(path):
    data = list()
    with open(path) as fid:
        next(fid)
        for line in fid:
            row = list()
            vals = line.split()
            for val in vals:
                row.append(float(val))
            data.append(row)
    fid.close()
    return np.asarray(data)


def read_label(path):
    vals = ""
    with open(path, 'r') as fid:
        for i, line in enumerate(fid):
            if i == 1:
                vals = line.split()
    fid.close()
    return vals[0], vals[1]


def create_db(features_dir):
    num_features = 9
    stats = list()
    y = list()
    for item in os.listdir(features_dir):
        # get only data files
        if not item.endswith('.txt'):
            continue
        m = read_data(features_dir + item)

        # cumulative_features = np.mean(m[:, 0:num_features], axis=0)
        cumulative_features = np.asarray(m.flatten())
        cur_stats = np.zeros(num_features*10)
        cur_stats[0:cumulative_features.shape[0]] = cumulative_features
        # cur_stats += cumulative_features
        stats.append(cur_stats)
        # stats.append(cumulative_features)
        if 'prevoiced' in item:
            y.append(1)
        else:
            y.append(0)

    x = np.asarray(stats)
    # avg = np.mean(x, axis=0)
    # std = np.std(x, axis=0)
    # x -= avg
    # x /= std
    y = np.asarray(y)
    np.nan_to_num(x)

    return x, y


def copy_files():
    for item in os.listdir('/Users/yossiadi/Datasets/vot/dmitrieva/neg_vot/fold_3/test/'):
        if item.endswith('.labels'):
            if 'prevoiced' in item:
                cmd = 'cp %s %s' % ('/Users/yossiadi/Datasets/vot/dmitrieva/features/prevoiced/labels/' + item,
                                    '/Users/yossiadi/Datasets/vot/dmitrieva/neg_vot/fold_3/test/labels/' + item)
            else:
                cmd = 'cp %s %s' % ('/Users/yossiadi/Datasets/vot/dmitrieva/features/voiced/labels/' + item,
                                    '/Users/yossiadi/Datasets/vot/dmitrieva/neg_vot/fold_3/test/labels/' + item)
                easy_call(cmd)


# ------------- MENU -------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Data creation for prevoicing detection detection")
    parser.add_argument("path_features", help="The path to features file")
    args = parser.parse_args()
    # copy_files()

    # run the script
    create_db(args.path_features)
