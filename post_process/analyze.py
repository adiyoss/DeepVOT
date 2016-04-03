from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def read_file(path):
    y = list()
    with open(path) as fid:
        for line in fid.readlines():
            curr_y = list()
            vals = line.split()
            for val in vals:
                curr_y.append(int(val))
            y.append(curr_y)
    fid.close()
    return y


y_all = read_file('labels.dat')
y_hat_all = read_file('pred.dat')

pos_neg_y = list()
pos_neg_y_hat = list()
for i, y in enumerate(y_all):
    if len(y) == 2:
        pos_neg_y.append(1)
    else:
        pos_neg_y.append(0)

    if y_hat_all[i][1] == -1:
        pos_neg_y_hat.append(1)
    else:
        pos_neg_y_hat.append(0)

print('Confusion Matrix: ')
print(confusion_matrix(pos_neg_y, pos_neg_y_hat))
print
print('Total Accuracy: %.3f' % accuracy_score(pos_neg_y, pos_neg_y_hat))
print('Precision: %.3f' % precision_score(pos_neg_y, pos_neg_y_hat))
print('Recall: %.3f' % recall_score(pos_neg_y, pos_neg_y_hat))
print('F1-Score: %.3f' % f1_score(pos_neg_y, pos_neg_y_hat))

cumulative_onset_pos = 0.0
cumulative_offset_pos = 0.0
count_pos = 0

cumulative_onset_neg = 0.0
cumulative_offset_neg = 0.0
count_neg = 0
for i, y in enumerate(y_all):
    if len(y) == 2:
        cumulative_onset_pos += abs(y[0] - y_hat_all[i][0])
        cumulative_offset_pos += abs(y[1] - y_hat_all[i][len(y_hat_all[i]) - 1])
        count_pos += 1
    if len(y) == 3:
        if y_hat_all[i][1] == -1:
            cumulative_onset_neg += abs(y[0] - y_hat_all[i][0])
            cumulative_offset_neg += abs(y[1] - y_hat_all[i][2])
            count_neg += 1
        else:
            cumulative_onset_neg += abs(y[0] - y_hat_all[i][0])
            cumulative_offset_neg += abs(y[1] - y_hat_all[i][1])
            count_neg += 1

print('Pos')
print('Average onset: %.3f' % (cumulative_onset_pos / count_pos))
print('Average offset: %.3f' % (cumulative_offset_pos / count_pos))
print('Neg')
print('Average onset: %.3f' % (cumulative_onset_neg / count_neg))
print('Average offset: %.3f' % (cumulative_offset_neg / count_neg))

cumulative_loss = 0.0
gamma_m = 4
gamma_0 = 100
count = 0
ms2 = 0
ms5 = 0
ms10 = 0
ms15 = 0
ms25 = 0
ms50 = 0
duration = 0
duration_hat = 0
flag = 0
for i, y in enumerate(y_all):
    if len(y) == 2 and y_hat_all[i][1] == -1:
        # duration = y[1] - y[0]
        # duration_hat = y_hat_all[i][2] - y_hat_all[i][0]
        # cumulative_loss += max(0, abs((y_hat_all[i][2] - y_hat_all[i][0]) - (y[1] - y[0]) - gamma_m))
        flag = 1
    elif len(y) == 3 and y_hat_all[i][1] != -1:
        duration = y[1] - y[0]
        duration_hat = y_hat_all[i][1] - y_hat_all[i][0]
        cumulative_loss += max(0, abs((y_hat_all[i][1] - y_hat_all[i][0]) - (y[1] - y[0]) - gamma_m))
        count += 1
    elif len(y) == 2 and y_hat_all[i][1] != -1:
        # duration = y[1] - y[0]
        # duration_hat = y_hat_all[i][1] - y_hat_all[i][0]
        # cumulative_loss += gamma_0
        flag = 1
    else:
        duration = y[1] - y[0]
        duration_hat = y_hat_all[i][2] - y_hat_all[i][0]
        cumulative_loss += gamma_0
        count += 1

    if flag == 0:
        diff = duration - duration_hat
        if diff < 2:
            ms2 += 1
        if diff < 5:
            ms5 += 1
        if diff < 10:
            ms10 += 1
        if diff < 15:
            ms15 += 1
        if diff < 25:
            ms25 += 1
        if diff < 50:
            ms50 += 1
    flag = 0

print(cumulative_loss / float(count))
print("==> 2ms > %.3f%%" % (100 * ms2 / float(count)))
print("==> 5ms > %.3f%%" % (100 * ms5 / float(count)))
print("==> 10ms > %.3f%%" % (100 * ms10 / float(count)))
print("==> 15ms > %.3f%%" % (100 * ms15 / float(count)))
print("==> 25ms > %.3f%%" % (100 * ms25 / float(count)))
print("==> 50ms > %.3f%%" % (100 * ms50 / float(count)))
