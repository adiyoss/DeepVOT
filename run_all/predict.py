import argparse
import os

import shutil

import sys

from extract_features import extract_features
from label2textgrid import create_text_grid
from lib import utils
from post_process import post_process
from run_backend import run


def predict(input_path, output_path, start_extract, end_extract):
    tmp_dir = 'tmp_files/'
    tmp_features = 'tmp.features'
    tmp_prob = 'tmp.prob'
    tmp_prediction = 'tmp.prediction'

    if not os.path.exists(input_path):
        print >>sys.stderr, "wav file does not exits"
        return

    length = utils.get_wav_file_length(input_path)
    feature_file = tmp_dir+tmp_features
    prob_file = tmp_dir+tmp_prob
    predict_file = tmp_dir+tmp_prediction

    # remove tmo dir if exists
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    print '\n1) Extracting features and classifying ...'
    extract_features(input_path, feature_file, start_extract, end_extract)
    run(feature_file, prob_file)
    print '\n3) Extract Durations ...'
    post_process(prob_file, predict_file)
    print '\n4) Writing TextGrid file to %s ...' % output_path
    create_text_grid(predict_file, output_path, length, float(start_extract))

    # remove leftovers
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="The path to the wav file")
    parser.add_argument("output_path", help="The path to save new text-grid file")
    parser.add_argument("start_extract", help="The time-stamp to start extracting features")
    parser.add_argument("end_extract", help="The time-stamp to end extracting features")
    args = parser.parse_args()

    # main function
    predict(args.input_path, args.output_path, args.start_extract, args.end_extract)
