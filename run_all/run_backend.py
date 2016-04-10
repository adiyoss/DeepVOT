import argparse
from lib import utils
import os


def run(features_path, output_path):
    f_abs_path = os.path.abspath(features_path)
    o_abs_path = os.path.abspath(output_path)
    os.chdir("lua_scripts/")
    cmd = 'th classify_multi_class.lua -input_file %s -output_file %s' % (f_abs_path, o_abs_path)
    utils.easy_call(cmd)
    os.chdir("..")


if __name__ == "__main__":
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("features_path", help="The path to the features file")
    parser.add_argument("output_path", help="The path to save the probabilities")
    args = parser.parse_args()

    # main function
    run(args.features_path, args.output_path)
