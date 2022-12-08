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


VISUALISE = False

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
    Y.append(1)
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
    Y.append(-1)
    # mfccAllChannels[i] = mfcc

    if VISUALISE:
        plt.imshow(mfcc, interpolation="nearest", origin="lower", aspect="auto")
        # plt.colorbar()
        plt.show()

for i in X:
    y_index = X.index(i)
    print(f"{Y[y_index]}: {i}")

# Split into train and test dataset
randIndices = []
for i in range(0, int(len(X) / 2)):
    # any random numbers from 0 to 1000
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
        X_train.append(X[i])
        Y_train.append(Y[i])

# Define model
config = {
    "K": 2,
    "Ninit": 2,
    "PVinit": 2,
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

# Compute the initial model parameters. Use the inital datapoints as initial prototype vectors
svm.compute(X_init, Y_init, X_pv, Y_pv)

print()
