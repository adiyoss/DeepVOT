import argparse
import os
import sys
import shutil as st
from lib import utils
from lib.textgrid import TextGrid


def measurement_features(audio_path, textgrid_path, output_path):
    # defines
    tmp_dir = 'tmp/'
    tmp_input = tmp_dir + 'tmp.input'
    tmp_label = tmp_dir + 'tmp.labels'
    label_suffix = '.labels'
    tmp_features = tmp_dir + 'tmp.features'
    tmp_file = tmp_dir + 'tmp.wav'
    epsilon = 0.001

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

                length = textgrid._TextGrid__tiers[0]._IntervalTier__intervals[2]._Interval__xmax

                onset = textgrid._TextGrid__tiers[0]._IntervalTier__intervals[1]._Interval__xmin
                offset = textgrid._TextGrid__tiers[0]._IntervalTier__intervals[1]._Interval__xmax

                start_extract = 0
                end_extract = min(offset + 0.08, length-epsilon)

                # =================== ACOUSTIC FEATURES =================== #
                # write labels
                label_file = output_path + item.replace('.wav', label_suffix)
                fid = open(label_file, 'w')
                fid.write('1 2\n')
                # fid.write('%s %s %s\n' % (
                #     int((voicing_start - start_extract) * 1000 + 1), int((voicing_end - start_extract) * 1000 + 1),
                #     int((release_end - start_extract) * 1000 + 1)))
                fid.write('%s %s %s\n' % (int(onset * 1000) + 1, int(offset * 1000) + 1, int(offset * 1000) + 4))
                fid.close()

                # creating the files
                input_file = open(tmp_features, 'wb')  # open the input file for the feature extraction
                features_file = open(tmp_input, 'wb')  # open file for the feature list path
                labels_file = open(tmp_label, 'wb')  # open file for the labels

                # write the data
                input_file.write(
                        '"' + tmp_file + '" ' + str('%.8f' % float(start_extract)) + ' ' + str(
                                float(end_extract)) + ' ' + str(
                                '%.8f' % float(onset)) + ' ' + str('%.8f' % float(offset)))
                features_file.write(output_path + item.replace('.wav', '.txt'))

                input_file.close()
                features_file.close()
                labels_file.close()

                command = "./sbin/VotFrontEnd2 %s %s %s" % (input_file.name, features_file.name, labels_file.name)
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
    measurement_features(args.audio_path, args.textgrid_path, args.output_path)
