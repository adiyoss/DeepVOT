import argparse
import os
import sys
import shutil as st
from lib.textgrid import TextGrid
from lib import utils


def neg_vot_creator(audio_path, textgrid_path, output_path, l):
    # defines
    tmp_dir = 'tmp/'
    tmp_input = tmp_dir + 'tmp.input'
    tmp_label = tmp_dir + 'tmp.labels'
    label_suffix = '.labels'
    tmp_features = tmp_dir + 'tmp.features'
    tmp_file = tmp_dir + 'tmp.wav'

    # validation
    if not os.path.exists(audio_path):
        print >> sys.stderr, 'Error: input path does not exists.'
        return
    if not os.path.exists(output_path):
        print 'output path does not exists, creating output directory.'
        os.mkdir(output_path)
    # create tmp dir
    if os.path.exists(tmp_dir):
        st.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    count = 0
    # loop over all the files in the input dir
    for item in os.listdir(audio_path):
        if item.endswith('.wav'):
            try:
                # convert to 16K 16bit
                cmd = 'sbin/sox %s -r 16000 -b 16 %s' % (audio_path + item, tmp_file)
                utils.easy_call(cmd)

                # parse the textgrid
                textgrid = TextGrid()
                textgrid.read(textgrid_path + item.replace('.wav', '.TextGrid'))
                release_start = textgrid._TextGrid__tiers[2]._IntervalTier__intervals[1]._Interval__xmin

                end_time = release_start
                if end_time - 0.1 < 0:
                    count += 1
                start_time = max(0, end_time - 0.1)

                # =================== ACOUSTIC FEATURES =================== #
                # write labels
                label_file = output_path + item.replace('.wav', label_suffix)
                fid = open(label_file, 'w')
                fid.write('%s\n' % str(l))
                fid.close()

                # creating the files
                input_file = open(tmp_features, 'wb')  # open the input file for the feature extraction
                features_file = open(tmp_input, 'wb')  # open file for the feature list path
                labels_file = open(tmp_label, 'wb')  # open file for the labels

                # write the data
                input_file.write(
                        '"' + tmp_file + '" ' + str('%.8f' % float(start_time)) + ' ' + str(
                                float(end_time)) + ' ' + str(
                                '%.8f' % float(start_time)) + ' ' + str('%.8f' % float(end_time)))
                features_file.write(output_path + item.replace('.wav', '.txt'))

                input_file.close()
                features_file.close()
                labels_file.close()

                command = "./sbin/VowelDurationFrontEnd %s %s %s" % (input_file.name, features_file.name, labels_file.name)
                utils.easy_call(command)

                # remove leftovers
                os.remove(tmp_input)
                os.remove(tmp_label)
                os.remove(tmp_features)
            except:
                print item
    st.rmtree(tmp_dir)

if __name__ == "__main__":
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="The path to the audio directory")
    parser.add_argument("textgrid_path", help="The path to the relevant textgrids")
    parser.add_argument("output_path", help="The path to output directory")
    args = parser.parse_args()

    # main function
    neg_vot_creator(args.audio_path, args.textgrid_path, args.output_path, 0)
