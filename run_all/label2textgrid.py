import os
import sys
import argparse
from lib.textgrid import *
from lib import utils

__author__ = 'yossiadi'


def create_text_grid(label_path, output_text_grid, length, start_extract):
    # defines
    msc_2_sec = 0.001

    # validation
    if not os.path.exists(label_path):
        print >>sys.stderr, "label file does not exits"
        return

    # read the label file and parse it
    fid = open(label_path)
    lines = fid.readlines()
    values = lines[0].split()
    fid.close()

    # create the TextGrid file and save it
    if len(values) == 3 and int(values[1]) == -1:    
        onset = values[0]
        offset = values[2]
        text_grid = TextGrid()

        vot_tier = IntervalTier(name='vot', xmin=0.0, xmax=float(length))
        vot_tier.append(Interval(0, float(onset)*msc_2_sec + start_extract, ""))
        vot_tier.append(Interval(float(onset)*msc_2_sec + start_extract, float(offset)*msc_2_sec + start_extract, "vot"))
        vot_tier.append(Interval(float(offset)*msc_2_sec + start_extract, float(length), ""))

        text_grid.append(vot_tier)
        text_grid.write(output_text_grid)
    
    if len(values) == 3 and int(values[1]) != -1:        
        prevoiced = values[0]
        onset = values[1]
        offset = values[2]
        text_grid = TextGrid()

        vot_tier = IntervalTier(name='vot', xmin=0.0, xmax=float(length))
        vot_tier.append(Interval(0, float(prevoiced)*msc_2_sec + start_extract, ""))
        vot_tier.append(Interval(float(prevoiced)*msc_2_sec + start_extract, float(onset)*msc_2_sec + start_extract, "prevoicing"))
        vot_tier.append(Interval(float(onset)*msc_2_sec + start_extract, float(offset)*msc_2_sec + start_extract, "vot"))
        vot_tier.append(Interval(float(offset)*msc_2_sec + start_extract, float(length), ""))

        text_grid.append(vot_tier)
        text_grid.write(output_text_grid)


if __name__ == "__main__":
    # the first argument is the label file path
    # the second argument is the wav file path
    # the third argument is the output path
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("label_filename", help="The label file")
    parser.add_argument("wav_filename", help="The wav file")
    parser.add_argument("output_text_grid", help="The output TextGrid file")
    args = parser.parse_args()

    # main function
    create_text_grid(args.label_filename, args.wav_filename, args.output_text_grid, 0.05)
