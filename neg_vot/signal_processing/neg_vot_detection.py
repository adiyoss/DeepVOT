from __future__ import print_function
import argparse
import os
import numpy as np
import sys

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics.classification import accuracy_score

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


def neg_vot_detection(features_dir, label_dir):
    y = list()
    y_hat = list()
    epsilon = 0.0000001
    for item in os.listdir(features_dir):
        # get only data files
        if not item.endswith('.txt'):
            continue
        # validate the existence of the label file
        label_path = label_dir+item.replace('.txt', '.labels')
        if not os.path.exists(label_path):
            print("Missing label to file:", item, file=sys.stderr)

        m = read_data(features_dir+item)
        l = read_label(label_path)

        if 'prevoiced' in item:
            y.append(1)
        else:
            y.append(0)

        flag = 0
        for i in range(int(l[0])):
            if m[:, 6][i] > epsilon:
                flag = 1
                y_hat.append(1)
                break
        if flag != 1:
            y_hat.append(0)

    print('Confusion Matrix: ')
    print(confusion_matrix(y, y_hat))
    print()
    print('Total Accuracy: %.3f' % accuracy_score(y, y_hat))
    print('Precision: %.3f' % precision_score(y, y_hat))
    print('Recall: %.3f' % recall_score(y, y_hat))
    print('F1-Score: %.3f' % f1_score(y, y_hat))


def copy_files():
    for item in os.listdir('/Users/yossiadi/Datasets/vot/dmitrieva/neg_vot/test/'):
        if item.endswith('.labels'):
            if 'prevoiced' in item:
                cmd = 'cp %s %s' % ('/Users/yossiadi/Datasets/vot/dmitrieva/features/prevoiced/labels/'+item, '/Users/yossiadi/Datasets/vot/dmitrieva/neg_vot/test/labels/'+item)
            else:
                cmd = 'cp %s %s' % ('/Users/yossiadi/Datasets/vot/dmitrieva/features/voiced/labels/'+item, '/Users/yossiadi/Datasets/vot/dmitrieva/neg_vot/test/labels/'+item)
            easy_call(cmd)

# ------------- MENU -------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Negative vot detection using signal processing only")
    parser.add_argument("path_features", help="The path to features file")
    parser.add_argument("path_labels", help="The path to features file")
    args = parser.parse_args()
    # copy_files()

    # run the script
    neg_vot_detection(args.path_features, args.path_labels)
