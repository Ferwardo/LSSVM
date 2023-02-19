import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from FE_funcs.feat_extract import FE
from LSSVM import LSSVM
from sklearn.metrics import f1_score, roc_auc_score


# from sklearn.metrics import f1_score, roc_auc_score


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


def get_data_set(device_number="0"):
    """
    Generates a training and test set for the selected device from each type with labels
    :param device_number: The device number to generate the dataset from.
    :return: A tuple with 4 matrices, one for the training set, one for the training labels, one for the test set
    and one for the test labels
    """
    X_train_all = {}
    Y_train_all = {}
    X_test_all = {}
    Y_test_all = {}

    for device_type in device_types:
        # Compute dataset
        normal = []
        for file in os.listdir(f"./dataset/federated_learning/{device_type}/id_0{device_number}/normal"):
            if file != "mfcc":
                normal.append(np.load(f"./dataset/federated_learning/{device_type}/id_0{device_number}/normal/" + file,
                                      allow_pickle=True))

        abnormal = []
        for file in os.listdir(f"./dataset/federated_learning/{device_type}/id_0{device_number}/abnormal"):
            if file != "mfcc":
                abnormal.append(
                    np.load(f"./dataset/federated_learning/{device_type}/id_0{device_number}/abnormal/" + file,
                            allow_pickle=True))

        # Dataset and labels
        X_normal = []
        Y_normal = []
        X_abnormal = []
        Y_abnormal = []
        Z_normal = []
        Z_abnormal = []

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

        X_train_all[device_type] = X_train
        Y_train_all[device_type] = Y_train
        X_test_all[device_type] = X_test
        Y_test_all[device_type] = Y_test

    return (X_train_all, Y_train_all, X_test_all, Y_test_all)


init_gpu(devices="1", v=1)
VISUALISE = False
class_labels = {
    "normal": 1,
    "abnormal": -1
}  # Either abnormal or normal
inverse_class_labels = {
    1: "normal",
    -1: "abnormal"
}
device_types = ["fan", "pump", "valve", "slider"]  # Different types of devices
server_devices = "0"  # This number selects the devices used on the server, NOT the total number of devices.

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

# Dataset generation and training of the server
X_train_all, Y_train_all, X_test_all, Y_test_all = get_data_set(server_devices)

# Initialise the server model.
server_model = LSSVM(config=config)

# Get prototype vectors
X_pv = []
Y_pv = []

for device_type in device_types:
    X_pv += X_train_all[device_type][0:int(server_model.config["PVinit"])]
    Y_pv += Y_train_all[device_type][0:int(server_model.config["PVinit"])]

    first_abnormal_index = Y_train_all[device_type].index(-1)
    X_pv += X_train_all[device_type][first_abnormal_index:first_abnormal_index + int(server_model.config["PVinit"])]
    Y_pv += Y_train_all[device_type][first_abnormal_index:first_abnormal_index + int(server_model.config["PVinit"])]

X_pv = tf.convert_to_tensor(X_pv)
Y_pv = tf.convert_to_tensor(X_pv, dtype=tf.float64)

# Get the initial dataset from each of the device types.
X_init = []
Y_init = []

for device_type in device_types:
    X_init += X_train_all[device_type][0:int(server_model.config["Ninit"])]
    Y_init += Y_train_all[device_type][0:int(server_model.config["Ninit"])]

    first_abnormal_index = Y_train_all[device_type].index(-1)
    X_init += X_train_all[device_type][first_abnormal_index:first_abnormal_index + int(server_model.config["Ninit"])]
    Y_init += Y_train_all[device_type][first_abnormal_index:first_abnormal_index + int(server_model.config["Ninit"])]

# Compute the initial model parameters. Use the inital datapoints as configured in "Ninit"
# and the initial prototype vectors as configured by "PVinit" the same datapoints can be used for both
X_init = tf.convert_to_tensor(X_init)
Y_init = tf.convert_to_tensor(Y_init, dtype=tf.float64)

server_model.compute(X_init, Y_init, X_pv, Y_pv)

results = {}

# Do a normal step for each device type with the server model
for device_type in device_types:
    server_model.normal(tf.convert_to_tensor(X_train_all[device_type]), tf.convert_to_tensor(Y_train_all[device_type]))

# Evaluate for each device type with the server model
for device_type in device_types:
    # variables for metrics calculated later
    right_number = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    Y_pred = []
    Y_pred_fx = []

    for i in range(0, len(X_test_all[device_type])):
        prediction, score = server_model.predict(tf.convert_to_tensor(X_test_all[device_type][i], dtype=tf.float64))
        Y_pred.append(prediction)
        Y_pred_fx.append(score)
        # print(
        # f"Right label: {inverse_class_labels[Y_test_all[device_type][i]]}. Predicted label: {inverse_class_labels[prediction]}")
        if Y_test_all[device_type][i] == prediction:
            right_number += 1
            if Y_test_all[device_type][i] == 1:
                true_positives += 1
        elif prediction == 1:
            false_positives += 1
        elif prediction == -1:
            false_negatives += 1

    Y_pred_fx = tf.convert_to_tensor(Y_pred_fx)
    Y_score = 1 / (1 + tf.math.exp(Y_pred_fx))
    Y_score = tf.squeeze(Y_score)

    results["server"][device_type]["accuracy"] = (right_number / len(X_test_all[device_type])) * 100
    results["server"][device_type]["precision"] = true_positives / (true_positives + false_positives)
    results["server"][device_type]["recall"] = true_positives / (true_positives + false_negatives)
    # f1_score = 2 * (precision * recall) / (precision + recall)
    results["server"][device_type]["f1_score"] = f1_score(Y_test_all[device_type], Y_pred)
    results["server"][device_type]["auc"] = roc_auc_score(Y_test_all[device_type], Y_score)

Beta_server = server_model.get_parameters(to_file=False)
Beta_envs = {}

# Dataset generation and training on each of the three environments
for i in ["2", "4", "6"]:
    X_train_all, Y_train_all, X_test_all, Y_test_all = get_data_set(i)

    env_model = LSSVM(Beta=Beta_server, config=config)

    # Do a normal step for each device type with the server model
    for device_type in device_types:
        server_model.normal(tf.convert_to_tensor(X_train_all[device_type]),
                            tf.convert_to_tensor(Y_train_all[device_type]))

    Beta_envs[i] = env_model.Beta

print("")
