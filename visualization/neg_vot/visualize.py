from __future__ import print_function
import argparse
import os
import numpy as np
import sys
from matplotlib import pyplot as plt


def ontype(event):
    if event.key == 'q':
        plt.close()
        sys.exit(0)
    else:
        return


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


def neg_vot_visualize(features_dir):
    num_features = 9
    stats = list()
    for item in os.listdir(features_dir):
        # get only data files
        if not item.endswith('.txt'):
            continue
        m = read_data(features_dir + item)
        cumulative_features = np.mean(m[:, 0:num_features], axis=0)
        cur_stats = np.zeros(num_features, )
        cur_stats += cumulative_features
        stats.append(cur_stats)

    np_stats = np.asarray(stats)
    avg = np.mean(np_stats, axis=0)
    std = np.std(np_stats, axis=0)
    np_stats -= avg
    np_stats /= std
    mx = np.max(np_stats)

    for item in os.listdir(features_dir):
        # get only data files
        if not item.endswith('.txt'):
            continue
        m = read_data(features_dir + item)

        cumulative_features = np.mean(m[:, 0:num_features], axis=0)
        cumulative_features -= avg
        cumulative_features /= std

        bar_width = 0.35
        opacity = 0.4
        error_config = {'ecolor': '0.3'}
        index = np.arange(num_features)
        fig = plt.figure(1, figsize=(15, 8))
        if 'prevoiced' in item:
            plt.title('Prevoiced')
        else:
            plt.title('Voiced')
        plt.figtext(0.1, 0.02, item, style='italic')
        fig.canvas.mpl_connect('key_press_event', ontype)
        plt.ylim(0, mx)
        plt.xticks(index + bar_width - 0.15, ('STE', 'TE', 'LE', 'HE', 'WE', 'AC', 'PIT', 'VOC', 'ZC'))
        plt.bar(index, np.abs(cumulative_features), bar_width,
                alpha=opacity,
                color='b',
                error_kw=error_config)
        plt.show()


# ------------- MENU -------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Negative vot detection using signal processing only")
    parser.add_argument("path_features", help="The path to features file")
    args = parser.parse_args()
    # copy_files()

    # run the script
    neg_vot_visualize(args.path_features)
