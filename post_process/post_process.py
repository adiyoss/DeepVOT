
import numpy as np



file_name = "full_prob.txt"
#P = np.loadtxt(file_name, delimiter=' ')  # exclude target column

with open(file_name) as f:
    lines = f.readlines()
f.close()

utterances = list()
new_utterance = list()
for line in lines:
    if line == '\n':
        if len(new_utterance) > 0:
            utterances.append(np.asarray(new_utterance, dtype=float))
        new_utterance = list()
        continue
    items = list()
    vals = line.split()
    for val in vals:
        items.append(float(val))
    new_utterance.append(items)


N_min = np.zeros(3, dtype=int)
N_max = np.zeros(3, dtype=int)
# silence
N_min[0] = 20
N_max[0] = 100
# prevoicing
N_min[1] = 40
N_max[1] = 120
# burst/release
N_min[2] = 2
N_max[2] = 100

for P in utterances:
    # number of frames
    N = P.shape[0]
    # with prevoicing
    D0_max = -100000
    n_best = np.zeros(3, dtype=int)
    for n0 in range(N_min[0], N-3):
        for n1 in range(n0 + N_min[1], N-2):
            for n2 in range(n1 + N_min[2], N-1):
                    D0 = np.sum(P[0:n0, 0]) + np.sum(P[(n0+1):n1, 1]) + np.sum(P[(n1+1):n2, 2]) + np.sum(P[(n2+1):N, 3])
                    if D0 > D0_max:
                        D0_max = D0
                        n_best[0] = n0
                        n_best[1] = n1
                        n_best[2] = n2
    # without prevoicing
    for n0 in range(N_min[0], N-3):
        for n2 in range(n0 + N_min[2], N-1):
            D0 = np.sum(P[0:n0, 0]) + np.sum(P[(n0+1):n2, 2]) + np.sum(P[(n2+1):N, 3])
            if D0 > D0_max:
                D0_max = D0
                n_best[0] = n0
                n_best[1] = -1
                n_best[2] = n2

    print D0_max, n_best
