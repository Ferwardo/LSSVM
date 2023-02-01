import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from FE_funcs.feat_extract import FE
from LSSVM import LSSVM


def init_gpu(devices="", v=2):
    if v == 2:  # tf2
        import tensorflow as tf
        import tensorflow.keras.backend as K
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # tf.config.experimental.set_visible_devices(physical_devices[devices], 'GPU')
                # gpus = tf.config.experimental.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        from tensorflow.python.framework.ops import disable_eager_execution
        # disable_eager_execution()
    else:  # tf1
        # os.environ["CUDA_VISIBLE_DEVICES"] = devices
        import tensorflow.keras.backend as K
        import tensorflow as tf
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        # tf.compat.v1.disable_eager_execution()
        tf.compat.v1.keras.backend.set_session(sess)
    return K


def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num, filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)

    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)

    return basis


init_gpu(devices="1", v=1)
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

# Model configuration parameters
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

X_pv_temp = []
Y_pv_temp = []

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
    for file in os.listdir("./dataset/id_00/normal"):
        if file != "mfcc":
            normal.append(np.load("./dataset/id_00/normal/" + file, allow_pickle=True))

    abnormal = []
    for file in os.listdir("./dataset/id_00/abnormal"):
        if file != "mfcc":
            abnormal.append(np.load("./dataset/id_00/abnormal/" + file, allow_pickle=True))

    # Dataset and labels
    X_normal = []
    Y_normal = []
    X_abnormal = []
    Y_abnormal = []
    Z_normal = []
    Z_abnormal = []
    samplerate = 44100

    counter = 0
    for mfcc in normal:
        counter += 1

        for channel in range(0, mfcc.shape[0]):
            means = []
            stds = []
            for filter in range(0, mfcc.shape[1]):
                means.append(mfcc[channel, filter].mean())
                stds.append(mfcc[channel, filter].std())

            X_normal.append(tuple(means + stds))
            Y_normal.append(class_labels["normal"])
            Z_normal.append(counter)

        if VISUALISE:
            for i in range(0, mfcc.shape[0]):
                plt.imshow(mfcc[i], interpolation="nearest", origin="lower", aspect="auto")
                # plt.colorbar()
                plt.show()

    abnormal_rate = 2
    index = -1
    for mfcc in abnormal:
        index += 1
        if index % abnormal_rate != 0:
            continue

        counter += 1

        for channel in range(0, mfcc.shape[0]):
            means = []
            stds = []
            for filter in range(0, mfcc.shape[1]):
                means.append(mfcc[channel, filter].mean())
                stds.append(mfcc[channel, filter].std())

            X_abnormal.append(tuple(means + stds))
            Y_abnormal.append(class_labels["abnormal"])
            Z_abnormal.append(counter)

        if VISUALISE:
            for i in range(0, mfcc.shape[0]):
                plt.imshow(mfcc[i], interpolation="nearest", origin="lower", aspect="auto")
                # plt.colorbar()
                plt.show()

    # Split into train and test dataset
    randIndices = []
    random.seed(256)  # use the same seed everytime, so we always get the same results

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    # For the normal training set
    for i in range(0, int(7 * (len(set(Z_normal)) / 10))):  # 70% of the data is used as training data
        randIndices.append(random.randint(0, len(set(Z_normal)) - 1))

    for i in randIndices:
        indices = [idx for idx, value in enumerate(Z_normal) if value == Z_normal[i]]
        for j in indices:
            X_train.append(X_normal[j])
            Y_train.append(Y_normal[j])

    for i in range(0, len(set(Z_normal))):
        if i not in randIndices:
            indices = [idx for idx, value in enumerate(Z_normal) if value == Z_normal[i]]
            for j in indices:
                X_test.append(X_normal[j])
                Y_test.append(Y_normal[j])

    X_pv_temp = X_pv_temp + X_train[0:int(config["PVinit"] / 2)]
    Y_pv_temp = Y_pv_temp + Y_train[0:int(config["PVinit"] / 2)]

    len_x_train_normal = len(X_train)
    # For the abnormal training set
    randIndices = []
    for i in range(0, int(7 * (len(set(Z_abnormal)) / 10))):  # 70% of the data is used as training data
        randIndices.append(random.randint(0, len(set(Z_abnormal)) - 1))

    for i in randIndices:
        # if i <= len(set(Z_abnormal)):
        indices = [idx for idx, value in enumerate(Z_abnormal) if value == Z_abnormal[i]]
        for j in indices:
            X_train.append(X_abnormal[j])
            Y_train.append(Y_abnormal[j])

    for i in range(0, len(set(Z_normal))):
        if i not in randIndices:
            # and i <= len(set(Z_abnormal)):

            indices = [idx for idx, value in enumerate(Z_abnormal) if value == Z_abnormal[i]]
            for j in indices:
                X_test.append(X_abnormal[j])
                Y_test.append(Y_abnormal[j])

    X_pv_temp = X_pv_temp + X_train[len_x_train_normal:len_x_train_normal + int(config["PVinit"] / 2)]
    Y_pv_temp = Y_pv_temp + Y_train[len_x_train_normal:len_x_train_normal + int(config["PVinit"] / 2)]

percentage = (Y_train.count(-1) / (Y_train.count(1) + Y_train.count(-1))) * 100
print(f"Amount of normal training samples: {Y_train.count(1)}")
print(f"Amount of abnormal training samples: {Y_train.count(-1)}")
print(f"Percentage of abnormal samples: {percentage}")

config["sigma"] = 2 * np.array(X_train).std()  # calculate the right kernel bandwidth.

# Define model
svm = LSSVM(config=config)

# Get the inital datapoints and prototypes
X_init = tf.convert_to_tensor(X_train[0:svm.config["Ninit"]])
Y_init = tf.convert_to_tensor(Y_train[0:svm.config["Ninit"]], dtype=tf.float64)

X_pv = tf.convert_to_tensor(X_pv_temp)
Y_pv = tf.convert_to_tensor(Y_pv_temp, dtype=tf.float64)

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
