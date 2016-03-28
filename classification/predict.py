import argparse


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


def predict(label_file, predictions_file):
    win_size = 8
    labels = read_file(label_file)
    raw_predictions = read_file(predictions_file)

    for row in raw_predictions:
        for i in xrange(len(row) - 1):
            if row[i] != row[i + 1]:
                flag = 0
                for j in xrange(1, min(win_size, len(row) - i)):
                    if row[i] == row[i + j]:
                        flag = j
                for j in xrange(1, flag):
                    row[i + j] = row[i]
    win_size = 6
    for row in raw_predictions:
        for i in xrange(len(row) - 1):
            if row[i] != row[i + 1]:
                flag = 0
                for j in xrange(1, min(win_size - 1, len(row) - i - 1)):
                    if row[i + 1] == row[i + j + 1]:
                        flag = j
                if flag < win_size - 2:
                    for j in xrange(flag + 1):
                        row[i + j + 1] = row[i]

    for row in raw_predictions:
        cur_val = row[0]
        for i in xrange(1, len(row) - 1):
            if row[i] < cur_val:
                row[i] = cur_val
            elif cur_val < row[i]:
                cur_val = row[i]

    predictions = list()
    for row in raw_predictions:
        curr_y = list()
        for i in xrange(len(row) - 1):
            if row[i] < row[i + 1]:
                curr_y.append(i)
        predictions.append(curr_y)

    onset = 0
    offset = 0
    count = 0
    p = 0
    for i, row in enumerate(predictions):
        count += 1
        if len(row) != len(labels[i]):
            p += 1
        if len(row) == 1:
            print(i)
            print(raw_predictions[i])
        else:
            onset += abs(row[0] - labels[i][0])
            offset += abs(row[1] - labels[i][1])

    onset /= float(count)
    offset /= float(count)
    p /= float(count)

    print('Average onset: ', onset)
    print('Average offset: ', offset)
    print('P: ', p)

if __name__ == "__main__":
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("label_file", help="The path to the label file")
    parser.add_argument("predictions_file", help="The path to the predictions file")
    args = parser.parse_args()

    # main function
    predict(args.label_file, args.predictions_file)
