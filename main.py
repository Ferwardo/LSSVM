import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from FE_funcs.feat_extract import FE
from LSSVM import LSSVM
from sklearn.metrics import f1_score, roc_auc_score
from scipy.optimize import curve_fit


def init_gpu(devices="", v=2):
    if v == 2:  # tf2
        import tensorflow as tf
        import tensorflow.keras.backend as K
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus, 'GPU')
                gpus = tf.config.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        from tensorflow.python.framework.ops import disable_eager_execution
        # disable_eager_execution()
    else:  # tf1
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        import tensorflow.keras.backend as K
        import tensorflow as tf
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        # tf.compat.v1.disable_eager_execution()
        tf.compat.v1.keras.backend.set_session(sess)
    return K


tf.config.list_physical_devices("GPU")  # With this my home gpu is seen, if left out it is not
init_gpu(devices="1", v=2)
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
    "Ninit": 4,
    "PVinit": 16,
    "M": 1500,
    "C": 10,
    "sigma": 20,
    "threshold_type": 'fxd',
    "threshold_minimum": 0.05,
}

X_pv_temp = []
Y_pv_temp = []

# Compute dataset
normal = []
for file in os.listdir("./dataset/id_00/normal"):
    if file != "mfcc":
        normal.append(np.load("./dataset/id_00/normal/" + file, allow_pickle=True).astype("float32"))

abnormal = []
for file in os.listdir("./dataset/id_00/abnormal"):
    if file != "mfcc":
        abnormal.append(np.load("./dataset/id_00/abnormal/" + file, allow_pickle=True).astype("float32"))

# # Get the dataset mean and standard deviation
mean_std_array = np.load("./dataset/federated_learning/mean_std.npy")
mean = mean_std_array[0]
std = mean_std_array[1]

# Normalise data
normal = (np.asarray(normal) - mean) / std
abnormal = (np.asarray(abnormal) - mean) / std

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

abnormal_rate = 1
index = -1
for mfcc in abnormal:
    index += 1
    if index % abnormal_rate != 0 and abnormal_rate != 1:
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

indices = []
for i in randIndices:
    if i in indices:
        continue
    indices = [idx for idx, value in enumerate(Z_normal) if value == Z_normal[i]]
    for j in indices:
        X_train.append(X_normal[j])
        Y_train.append(Y_normal[j])

indices = []
for i in range(0, len(set(Z_normal))):
    if i in indices:
        continue
    if i not in randIndices:
        indices = [idx for idx, value in enumerate(Z_normal) if value == Z_normal[i]]
        for j in indices:
            X_test.append(X_normal[j])
            Y_test.append(Y_normal[j])

# len_x_train_normal = len(X_train)

# For the abnormal training set
randIndices = []
for i in range(0, int(7 * (len(set(Z_abnormal)) / 10))):  # 70% of the data is used as training data
    randIndices.append(random.randint(0, len(set(Z_abnormal)) - 1))

indices = []
for i in randIndices:
    if i in indices:
        continue
    # if i <= len(set(Z_abnormal)):
    indices = [idx for idx, value in enumerate(Z_abnormal) if value == Z_abnormal[i]]
    for j in indices:
        # X_train = [X_abnormal[j]] + X_train
        # Y_train = [Y_abnormal[j]] + Y_train
        X_train.append(X_abnormal[j])
        Y_train.append(Y_abnormal[j])

indices = []
for i in range(0, len(set(Z_abnormal))):
    if i in indices:
        continue

    if i not in randIndices:
        # and i <= len(set(Z_abnormal)):

        indices = [idx for idx, value in enumerate(Z_abnormal) if value == Z_abnormal[i]]
        for j in indices:
            X_test.append(X_abnormal[j])
            Y_test.append(Y_abnormal[j])

first_abnormal_index = Y_train.index(-1)
X_pv_temp = X_pv_temp + X_train[first_abnormal_index:(first_abnormal_index + int(config["PVinit"] * 4))]
Y_pv_temp = Y_pv_temp + Y_train[first_abnormal_index:(first_abnormal_index + int(config["PVinit"] * 4))]

percentage = (Y_train.count(-1) / (Y_train.count(1) + Y_train.count(-1))) * 100
print(f"Amount of normal training samples: {Y_train.count(1)}")
print(f"Amount of abnormal training samples: {Y_train.count(-1)}")
print(f"Percentage of abnormal samples: {percentage}")

config["sigma"] = 2 * np.array(X_train).std()  # calculate the right kernel bandwidth.

# Define model
svm = LSSVM(config=config)

# Get the inital datapoints and prototypes
X_init = tf.convert_to_tensor(X_train[0:svm.config["Ninit"] * 4] + X_train[-svm.config["Ninit"] * 4:])
Y_init = tf.convert_to_tensor(Y_train[0:svm.config["Ninit"] * 4] + Y_train[-svm.config["Ninit"] * 4:],
                              dtype=tf.float64)

X_pv = tf.convert_to_tensor(X_pv_temp)
Y_pv = tf.convert_to_tensor(Y_pv_temp, dtype=tf.float64)

# Compute the initial model parameters. Use the inital datapoints as configured in "Ninit"
# and the initial prototype vectors as configured by "PVinit" the same datapoints can be used for both
svm.compute(X_init, Y_init, X_pv, Y_pv)

# Do a normal step
svm.normal(tf.convert_to_tensor(X_train), tf.convert_to_tensor(Y_train))

# Do predictions for all test data

# variables for metrics calculated later
right_number = 0
true_positives = 0
false_positives = 0
false_negatives = 0
Y_pred = []
Y_pred_fx = []

for i in range(0, len(X_test)):
    prediction, score = svm.predict(tf.convert_to_tensor(X_test[i], dtype=tf.float64))
    Y_pred.append(prediction)
    Y_pred_fx.append(score)
    # print(f"Right label: {inverse_class_labels[Y_test[i]]}. Predicted label: {inverse_class_labels[prediction]}")
    if Y_test[i] == prediction:
        right_number += 1
        if Y_test[i] == 1:
            true_positives += 1
    elif prediction == 1:
        false_positives += 1
    elif prediction == -1:
        false_negatives += 1

Y_pred_fx = tf.squeeze(Y_pred_fx)
accuracy = (right_number / len(X_test)) * 100
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
# f1_score = 2 * (precision * recall) / (precision + recall)
f1_score = f1_score(Y_test, Y_pred)
auc = roc_auc_score(Y_test, Y_pred_fx)

print(f"Testing the svm with {svm.config['PVinit']} prototype vectors for each channel.")
print("================================================")
print(f"Accuracy for current test set: {str(round(accuracy, 2))}%")
print(f"Precision for current test set: {str(round(precision, 4))}")
print(f"Recall for current test set: {str(round(recall, 4))}")
print(f"F1 score for current test set: {str(round(f1_score, 4))}")
print(f"AUC for current test set: {str(round(auc, 4))}")

# print("Get the parameters for the federated learning")
svm.get_federated_learning_params(as_json=True, to_file=True)
