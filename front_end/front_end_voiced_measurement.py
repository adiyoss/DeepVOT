import argparse
import os
import sys
import shutil as st
from subprocess import call
from lib.textgrid import TextGrid


# run system commands
def easy_call(command):
    try:
        call(command, shell=True)
    except Exception as exception:
        print "Error: could not execute the following"
        print ">>", command
        print type(exception)  # the exception instance
        print exception.args  # arguments stored in .args
        exit(-1)


def measurement_features(audio_path, textgrid_path, output_path):
    # defines
    tmp_dir = 'tmp/'
    tmp_input = tmp_dir + 'tmp.input'
    tmp_label = tmp_dir + 'tmp.labels'
    label_suffix = '.labels'
    tmp_features = tmp_dir + 'tmp.features'
    tmp_file = tmp_dir + 'tmp.wav'
    gap_start = 0.02
    # gap_start = 0.1
    gap_end = 0.02

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
                easy_call(cmd)

                # parse the textgrid
                textgrid = TextGrid()
                textgrid.read(textgrid_path + item.replace('.wav', '.TextGrid'))
                release_start = textgrid._TextGrid__tiers[2]._IntervalTier__intervals[1]._Interval__xmin
                release_end = textgrid._TextGrid__tiers[2]._IntervalTier__intervals[1]._Interval__xmax

                voicing_start = textgrid._TextGrid__tiers[5]._IntervalTier__intervals[1]._Interval__xmin
                voicing_end = textgrid._TextGrid__tiers[5]._IntervalTier__intervals[1]._Interval__xmax

                # onset = min(release_start, voicing_start)
                # offset = max(release_end, voicing_end)

                onset = release_start
                offset = release_end

                start_extract = onset - gap_start
                end_extract = offset + gap_end

                # =================== ACOUSTIC FEATURES =================== #
                # write labels
                label_file = output_path + item.replace('.wav', label_suffix)
                fid = open(label_file, 'w')
                fid.write('1 2\n')
                # fid.write('%s %s %s\n' % (
                #     int((voicing_start - start_extract) * 1000 + 1), int((voicing_end - start_extract) * 1000 + 1),
                #     int((release_end - start_extract) * 1000 + 1)))
                fid.write('%s %s %s %s\n' % (
                    int((release_start - start_extract) * 1000 + 1), int((release_end - start_extract) * 1000 + 1),
                    int((voicing_start - start_extract) * 1000 + 1), int((voicing_end - start_extract) * 1000 + 1)))
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
                easy_call(command)

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

    # # loop over all the files in the input dir
    # for item in os.listdir(args.audio_path):
    #     if item.endswith('.wav'):
    #         if not os.path.exists(args.textgrid_path+item.replace('.wav', '.TextGrid')):
    #             print item

    # /Users/yossiadi/Datasets/vot/dmitrieva/prevoiced/ /Users/yossiadi/Datasets/vot/dmitrieva/prevoiced/ /Users/yossiadi/Datasets/vot/dmitrieva/features/prevoiced/
