from __future__ import print_function

import argparse

import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from data import create_db
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adagrad
from keras.utils import np_utils

np.random.seed(1337)  # for reproducibility


def train(train_path, test_path, model_path):
    # params
    batch_size = 32
    nb_epoch = 50
    in_dim = 90
    out_dim = 2
    hidden_size = 512

    # the data, shuffled and split between train and test sets
    (X_train, y_train) = create_db(train_path)
    (X_test, y_test) = create_db(test_path)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, out_dim)
    Y_test = np_utils.to_categorical(y_test, out_dim)

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(in_dim,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(hidden_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(hidden_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(out_dim))
    model.add(Activation('softmax'))

    opt = Adagrad()
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    model_save = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=False,
                                 mode='auto')

    model.fit(X_train, Y_train,
              batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=2, callbacks=[model_save])

    y_hat = np.argmax(model.predict(X_test), axis=1)

    print('Confusion Matrix: ')
    print(confusion_matrix(y_test, y_hat))
    print()
    print('Total Accuracy: %.3f' % accuracy_score(y_test, y_hat))
    print('Precision: %.3f' % precision_score(y_test, y_hat))
    print('Recall: %.3f' % recall_score(y_test, y_hat))
    print('F1-Score: %.3f' % f1_score(y_test, y_hat))


# ------------- MENU -------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Train neural network for prevoicing detection")
    parser.add_argument("--train_path", help="The path to the train features files",
                        default='/Users/yossiadi/Datasets/vot/dmitrieva/neg_vot/10ms_features/fold_2/train/')
    parser.add_argument("--test_path", help="The path to the test features files",
                        default='/Users/yossiadi/Datasets/vot/dmitrieva/neg_vot/10ms_features/fold_2/test/')
    parser.add_argument("--model_path", help="The path to save the model",
                        default='model/fold.2.model.512.net')
    args = parser.parse_args()

    # run the script
    train(args.train_path, args.test_path, args.model_path)
