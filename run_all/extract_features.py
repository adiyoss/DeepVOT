import argparse
import os
import sys
import shutil as st
from lib import utils


def extract_features(audio_path, output_path, start_extract, end_extract):
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
    # create tmp dir
    if os.path.exists(tmp_dir):
        st.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    # loop over all the files in the input dir
    if audio_path.endswith('.wav'):
        try:
            # convert to 16K 16bit
            cmd = 'sbin/sox %s -r 16000 -b 16 %s' % (audio_path, tmp_file)
            utils.easy_call(cmd)

            onset = (float(start_extract) + float(end_extract)) / 2
            offset = (float(start_extract) + float(end_extract)) / 2

            # =================== ACOUSTIC FEATURES =================== #
            # # write labels
            # label_file = audio_path.replace('.wav', label_suffix)
            # fid = open(label_file, 'w')
            # fid.write('1 2\n')
            # fid.write('%s %s %s\n' % (str(1), str(1), str(1)))
            # fid.close()

            # creating the files
            input_file = open(tmp_features, 'wb')  # open the input file for the feature extraction
            features_file = open(tmp_input, 'wb')  # open file for the feature list path
            labels_file = open(tmp_label, 'wb')  # open file for the labels

            # write the data
            input_file.write(
                    '"' + tmp_file + '" ' + str('%.8f' % float(start_extract)) + ' ' + str(
                            float(end_extract)) + ' ' + str(
                            '%.8f' % float(onset)) + ' ' + str('%.8f' % float(offset)))
            features_file.write(output_path.replace('.wav', '.txt'))

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
            print audio_path
    st.rmtree(tmp_dir)


if __name__ == "__main__":
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="The path to the audio file")
    parser.add_argument("output_path", help="The path to save the features")
    parser.add_argument("start_extract", help="The time-stamp to start extracting features")
    parser.add_argument("end_extract", help="The time-stamp to end extracting features")
    args = parser.parse_args()

    # main function
    extract_features(args.audio_path, args.output_path, args.start_extract, args.end_extract)
