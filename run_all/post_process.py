import argparse
import numpy as np
import os


def post_process(prob_file, output_path):
    prob_file = os.path.abspath(prob_file)
    output_path = os.path.abspath(output_path)

    ##############################
    # parsing the probability file
    with open(prob_file) as f:
        lines = f.readlines()
    f.close()

    utter = list()
    for line in lines:
        items = list()
        vals = line.split()
        for val in vals:
            items.append(float(val))
        utter.append(items)
    N_min = np.zeros(3, dtype=int)
    N_max = np.zeros(3, dtype=int)
    ##############################

    # silence
    N_min[0] = 20
    N_max[0] = 100

    # prevoicing
    N_min[1] = 40
    N_max[1] = 120

    # burst/release
    N_min[2] = 2
    N_max[2] = 100

    predictions = list()
    P = np.asarray(utter)
    
    # number of frames
    N = P.shape[0]

    # with prevoicing
    D0_max = -100000
    n_best = np.zeros(3, dtype=int)
    for n0 in range(N_min[0], N - 3):
        for n1 in range(n0 + N_min[1], N - 2):
            for n2 in range(n1 + N_min[2], N - 1):
                D0 = np.sum(P[0:n0, 0]) + np.sum(P[(n0 + 1):n1, 1]) + np.sum(P[(n1 + 1):n2, 2]) + np.sum(
                        P[(n2 + 1):N, 3])
                if D0 > D0_max:
                    D0_max = D0
                    n_best[0] = n0
                    n_best[1] = n1
                    n_best[2] = n2

    # without prevoicing
    for n0 in range(N_min[0], N - 3):
        for n2 in range(n0 + N_min[2], N - 1):
            D0 = np.sum(P[0:n0, 0]) + np.sum(P[(n0 + 1):n2, 2]) + np.sum(P[(n2 + 1):N, 3])
            if D0 > D0_max:
                D0_max = D0
                n_best[0] = n0
                n_best[1] = -1
                n_best[2] = n2

    predictions.append(n_best)

    with open(output_path, 'w') as fid:
        for p in predictions:
            fid.write(str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + '\n')
    fid.close()


if __name__ == "__main__":
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("prob_file", help="The path to the probabilities file that were generated from the nn")
    parser.add_argument("output_path", help="The path to save the label")
    args = parser.parse_args()

    # main function
    post_process(args.prob_file, args.output_path)
