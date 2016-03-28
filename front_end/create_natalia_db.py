import argparse
import os
from lib import utils
from lib.textgrid import TextGrid, IntervalTier, Interval


def create_db(audio_path, textgrid_path, output_path):
    voiced_path = output_path+'voiced/'
    prevoiced_path = output_path+'prevoiced/'
    c = 0
    for item in os.listdir(audio_path):
        tg_file_path = textgrid_path + item.replace('.wav', '.TextGrid')
        if item.endswith('.wav') and os.path.exists(tg_file_path):
            c += 1
            print('Processing item: %d, file name: %s' % (c, item))

            tg = TextGrid()
            tg.read(tg_file_path)
            tier = tg._TextGrid__tiers[1]

            print('Creating output dirs ...')
            # create the relevant dirs
            if not os.path.exists(voiced_path):
                os.mkdir(voiced_path)
            if not os.path.exists(prevoiced_path):
                os.mkdir(prevoiced_path)

            for i, interval in enumerate(tier._IntervalTier__intervals):
                if 'ne' in interval._Interval__mark:
                    # gap = 0.1
                    gap = 0.02
                    start_vot = interval._Interval__xmin
                    end_vot = interval._Interval__xmax
                    start = start_vot - gap
                    end = end_vot + gap
                    output_name = os.path.abspath(prevoiced_path) + '/' + str(i) + '_' + item
                    utils.crop_wav(os.path.abspath(audio_path + item), start, end,
                                   output_name)

                    # write the text grid
                    length = end - start
                    new_tg = TextGrid()
                    vot_tier = IntervalTier(name='VOT', xmin=0.0, xmax=float(length))
                    vot_tier.append(Interval(0, start_vot - start, ""))
                    vot_tier.append(Interval(start_vot - start, end_vot - start, "neg"))
                    vot_tier.append(Interval(end_vot - start, float(length), ""))

                    new_tg.append(vot_tier)
                    new_tg.write(output_name.replace('.wav', '.TextGrid'))

                elif 'v' in interval._Interval__mark:
                    gap = 0.02
                    start_vot = interval._Interval__xmin
                    end_vot = interval._Interval__xmax
                    start = start_vot - gap
                    end = end_vot + gap
                    output_name = os.path.abspath(voiced_path) + '/' + str(i) + '_' + item
                    utils.crop_wav(os.path.abspath(audio_path + item), start, end,
                                   output_name)

                    # write the text grid
                    length = end - start
                    new_tg = TextGrid()
                    vot_tier = IntervalTier(name='VOT', xmin=0.0, xmax=float(length))
                    vot_tier.append(Interval(0, start_vot - start, ""))
                    vot_tier.append(Interval(start_vot - start, end_vot - start, "v"))
                    vot_tier.append(Interval(end_vot - start, float(length), ""))

                    new_tg.append(vot_tier)
                    new_tg.write(output_name.replace('.wav', '.TextGrid'))
            print('Done.')
    return None


if __name__ == "__main__":
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="The path to the audio directory")
    parser.add_argument("textgrid_path", help="The path to the relevant textgrids")
    parser.add_argument("output_path", help="The path to output directory")
    args = parser.parse_args()

    # main function
    create_db(args.audio_path, args.textgrid_path, args.output_path)
