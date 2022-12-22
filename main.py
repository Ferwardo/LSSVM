import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from FE_funcs.feat_extract import FE
from LSSVM import LSSVM


def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num, filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)

    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)

    return basis


MIMII = True
VISUALISE = False
class_labels = {
    "normal": 1,
    "abnormal": -1
}
inverse_class_labels = {
    1: "normal",
    -1: "abnormal"
}

if not MIMII:
    # Compute dataset
    normal = []
    for file in os.listdir("./dataset/normal"):
        normal.append(np.load("./dataset/normal/" + file, allow_pickle=True))

    abnormal = []
    for file in os.listdir("./dataset/abnormal"):
        abnormal.append(np.load("./dataset/abnormal/" + file, allow_pickle=True))

    # Dataset and labels
    X = []
    Y = []
    samplerate = 44100

    # Define feature extraction parameters
    featconf = {}
    featconf['dcRemoval'] = 'hpf'
    featconf['samFreq'] = samplerate
    featconf['lowFreq'] = 0
    featconf['highFreq'] = featconf['samFreq'] / 2
    featconf['stepSize_ms'] = 10
    featconf['frameSize_ms'] = 32
    featconf['melSize'] = 64

    featextract = FE(featconf)

    normal_temp = []
    for data in normal:
        # for i in range(0, len(data[1])):
        logmelframes = featextract.fe_transform(data[1])

        dct_filters = dct(20, int(np.floor(((data[0] - (featconf['frameSize'] - 1) - 1) / featconf['stepSize']) + 1)))

        mfcc = np.dot(dct_filters, logmelframes)
        X.append((mfcc.mean(), mfcc.std()))
        Y.append(class_labels["normal"])
        # mfccAllChannels[i] = mfcc

        if VISUALISE:
            plt.imshow(mfcc, interpolation="nearest", origin="lower", aspect="auto")
            # plt.colorbar()
            plt.show()

    for data in abnormal:
        # for i in range(0, len(data[1])):
        logmelframes = featextract.fe_transform(data[1])

        dct_filters = dct(20, int(np.floor(((data[0] - (featconf['frameSize'] - 1) - 1) / featconf['stepSize']) + 1)))

        mfcc = np.dot(dct_filters, logmelframes)
        X.append((mfcc.mean(), mfcc.std()))
        Y.append(class_labels["abnormal"])
        # mfccAllChannels[i] = mfcc

        if VISUALISE:
            plt.imshow(mfcc, interpolation="nearest", origin="lower", aspect="auto")
            # plt.colorbar()
            plt.show()

    # for i in X:
    #     y_index = X.index(i)
    #     print(f"{Y[y_index]}: {i}")

    # Split into train and test dataset
    randIndices = []
    random.seed(256)  # use the same seed everytime so we always get the same results
    for i in range(0, int(7 * (len(X) / 10))):  # 70% of the data is used as training data
        randIndices.append(random.randint(0, len(X) - 1))

    X_train = []
    Y_train = []

    for i in randIndices:
        X_train.append(X[i])
        Y_train.append(Y[i])

    X_test = []
    Y_test = []

    for i in range(0, len(X)):
        if i not in randIndices:
            X_test.append(X[i])
            Y_test.append(Y[i])
else:
    # Compute dataset
    normal = []
    for file in os.listdir("../MIMII/dataset/fan/id_00/normal"):
        if file != "mfcc":
            normal.append(np.load("../MIMII/dataset/fan/id_00/normal/" + file, allow_pickle=True))

    abnormal = []
    for file in os.listdir("../MIMII/dataset/fan/id_00/abnormal"):
        if file != "mfcc":
            abnormal.append(np.load("../MIMII/dataset/fan/id_00/abnormal/" + file, allow_pickle=True))

    normal = normal[0:len(abnormal)]  # This is so we get as much normal as abnormal samples.
    # Otherwise the classes are imbalanced
    # Dataset and labels
    X = []
    Y = []
    samplerate = 44100

    for mfcc in normal:
        MeanAndSTD = []
        for channel in range(0, mfcc.shape[0]):
            MeanAndSTD.append(mfcc[channel].mean())
            MeanAndSTD.append(mfcc[channel].std())
        X.append(tuple(MeanAndSTD))
        Y.append(class_labels["normal"])
        # mfccAllChannels[i] = mfcc

        if VISUALISE:
            for i in range(0, mfcc.shape[0]):
                plt.imshow(mfcc[i], interpolation="nearest", origin="lower", aspect="auto")
                # plt.colorbar()
                plt.show()

    for mfcc in abnormal:
        MeanAndSTD = []
        for channel in range(0, mfcc.shape[0]):
            MeanAndSTD.append(mfcc[channel].mean())
            MeanAndSTD.append(mfcc[channel].std())
        X.append(tuple(MeanAndSTD))
        Y.append(class_labels["abnormal"])
        # mfccAllChannels[i] = mfcc

        if VISUALISE:
            for i in range(0, mfcc.shape[0]):
                plt.imshow(mfcc[i], interpolation="nearest", origin="lower", aspect="auto")
                # plt.colorbar()
                plt.show()

    # Split into train and test dataset
    randIndices = []
    random.seed(256)  # use the same seed everytime so we always get the same results
    for i in range(0, int(7 * (len(X) / 10))):  # 70% of the data is used as training data
        randIndices.append(random.randint(0, len(X) - 1))

    X_train = []
    Y_train = []

    for i in randIndices:
        X_train.append(X[i])
        Y_train.append(Y[i])

    X_test = []
    Y_test = []

    for i in range(0, len(X)):
        if i not in randIndices:
            X_test.append(X[i])
            Y_test.append(Y[i])

percentage = (Y_train.count(-1) / (Y_train.count(1) + Y_train.count(-1))) * 100
print(f"Amount of normal training samples: {Y_train.count(1)}")
print(f"Amount of abnormal training samples: {Y_train.count(-1)}")
print(f"Percentage of abnormal samples: {percentage}")

# Define model
config = {
    "K": 2,
    "Ninit": 16,
    "PVinit": 32,
    "M": 1500,
    "C": 10,
    "sigma": 20,
    "threshold_type": 'fxd',
    "threshold_minimum": 0.05,
}
svm = LSSVM(config=config)

# Get the inital datapoints and prototypes
X_init = tf.convert_to_tensor(X_train[0:svm.config["Ninit"]])
Y_init = tf.convert_to_tensor(Y_train[0:svm.config["Ninit"]], dtype=tf.float64)

X_pv = tf.convert_to_tensor(X_train[0:svm.config["PVinit"]])
Y_pv = tf.convert_to_tensor(Y_train[0:svm.config["PVinit"]], dtype=tf.float64)

# Compute the initial model parameters. Use the inital datapoints as configured in "Ninit"
# and the initial prototype vectors as configured by "PVinit" the same datapoints can be used for both
svm.compute(X_init, Y_init, X_pv, Y_pv)

# Do a normal step
svm.normal(tf.convert_to_tensor(X_train), tf.convert_to_tensor(Y_train))

# Do predictions for all test data
print(f"Testing the svm with {svm.config['PVinit']} prototype vectors")
print("================================================")

right_number = 0
for i in range(0, len(X_test)):
    prediction = svm.predict(tf.convert_to_tensor(X_test[i], dtype=tf.float64))
    print(f"Right label: {inverse_class_labels[Y_test[i]]}. Predicted label: {inverse_class_labels[prediction]}")
    if Y_test[i] == prediction:
        right_number += 1

accuracy = (right_number / len(X_test)) * 100

print("================================================")
print(f"Accuracy for current test set: {accuracy}%\n")

# print("Get the parameters for the federated learning")
svm.get_federated_learning_params(as_json=True, to_file=True)
