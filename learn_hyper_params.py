import os
from random import random

import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from LSSVM import LSSVM
from sklearn.model_selection import GridSearchCV


def init_gpu(devices="", v=2):
    if v == 2:  # tf2
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.set_visible_devices(gpus, 'GPU')
                gpus = tf.config.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        # from tensorflow.python.framework.ops import disable_eager_execution
        # disable_eager_execution()
    else:  # tf1
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        # tf.compat.v1.disable_eager_execution()
        tf.compat.v1.keras.backend.set_session(sess)
    return K


tf.config.list_physical_devices("GPU")  # With this my home gpu is seen, if left out it is not
init_gpu(devices="1", v=2)
VISUALISE = False
with_subsampling = True
class_labels = {
    "normal": 1,
    "abnormal": -1
}
inverse_class_labels = {
    1: "normal",
    -1: "abnormal"
}
device_types = ["fan",
                "pump",
                "valve",
                "slider"
                ]  # Different types of devices

# Load data
normal = []
abnormal = []
for device_type in device_types:
    for file in os.listdir(f"./dataset/federated_learning/{device_type}/id_00/normal"):
        try:
            if file != "mfcc":
                normal.append(
                    np.load(f"./dataset/federated_learning/{device_type}/id_00/normal/" + file,
                            allow_pickle=True).astype("float32"))
        except:
            print(f"./dataset/federated_learning/{device_type}/id_00/normal/{file} not loaded")

    for file in os.listdir(f"./dataset/federated_learning/{device_type}/id_00/abnormal"):
        try:
            if file != "mfcc":
                abnormal.append(
                    np.load(f"./dataset/federated_learning/{device_type}/id_00/abnormal/" + file,
                            allow_pickle=True).astype("float32"))
        except:
            print(f"./dataset/federated_learning/{device_type}/id_00/abnormal/{file}")

# Normalise data
mean_std_array = np.load("./dataset/federated_learning/mean_std.npy")
mean = mean_std_array[0]
std = mean_std_array[1]

normal = (np.asarray(normal) - mean) / std
abnormal = (np.asarray(abnormal) - mean) / std

# Dataset and labels
X_normal = []
Y_normal = []
X_abnormal = []
Y_abnormal = []
Z_normal = []
Z_abnormal = []

counter = 0
subsampling_cnt = 0
for mfcc in normal:
    counter += 1

    if with_subsampling:
        subsampling_cnt += 1
        if subsampling_cnt != 1:
            if subsampling_cnt >= 10:
                subsampling_cnt = 0
            continue

    for channel in range(0, mfcc.shape[0]):
        means = []
        stds = []
        for filter in range(0, mfcc.shape[2]):
            means.append(mfcc[channel, :, filter].mean())
            stds.append(mfcc[channel, :, filter].std())

        X_normal.append(tuple(means + stds))
        Y_normal.append(class_labels["normal"])
        Z_normal.append(counter)

    if VISUALISE:
        for i in range(0, mfcc.shape[0]):
            plt.imshow(mfcc[i], interpolation="nearest", origin="lower", aspect="auto")
            # plt.colorbar()
            plt.show()

# Set the normal variable to be garbage collected as it isn't needed anymore
del normal

# abnormal_rate = 1
# index = -1
subsampling_cnt = 0
for mfcc in abnormal:
    # index += 1
    # if index % abnormal_rate != 0 and abnormal_rate != 1:
    #     continue

    counter += 1

    if with_subsampling:
        subsampling_cnt += 1
        if subsampling_cnt != 1:
            if subsampling_cnt >= 10:
                subsampling_cnt = 0
            continue

    for channel in range(0, mfcc.shape[0]):
        means = []
        stds = []
        for filter in range(0, mfcc.shape[2]):
            means.append(mfcc[channel, :, filter].mean())
            stds.append(mfcc[channel, :, filter].std())

    X_abnormal.append(tuple(means + stds))
    Y_abnormal.append(class_labels["abnormal"])
    Z_abnormal.append(counter)

    if VISUALISE:
        for i in range(0, mfcc.shape[0]):
            plt.imshow(mfcc[i], interpolation="nearest", origin="lower", aspect="auto")
            # plt.colorbar()
            plt.show()

# Set the abnormal variable to be garbage collected as it isn't needed anymore
del abnormal

X = np.concatenate([X_normal, X_abnormal])
Y = np.concatenate([Y_normal, Y_abnormal])

C_range = np.logspace(-2, 10, 13)  # taken from the examples of sklearn
sigma_range = np.logspace(-9, 3, 13)
param_grid = dict(C=C_range, sigma=sigma_range)
grid = GridSearchCV(estimator=LSSVM(), scoring="accuracy", param_grid=param_grid)
grid.fit(X, Y)

print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)
