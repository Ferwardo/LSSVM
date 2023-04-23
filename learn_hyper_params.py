import os
import random
from multiprocessing import Process

import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from LSSVM import LSSVM
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit


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


def get_data_set(device_number="0", with_subsampling=False, mean_calc_needed=False):
    """
    Generates a training and test set for the selected device from each type with labels
    :param mean_calc_needed: Whether a calculation of the mean and standard deviation is needed for the dataset.
    :param device_number: The device number to generate the dataset from.
    :param with_subsampling: Subsamples the dataset for faster training while debugging.
    :return: A tuple with 4 matrices/vectors, one for the training set, one for the training labels, one for the test set
    and one for the test labels
    """
    X_train_all = {}
    Y_train_all = {}
    X_test_all = {}
    Y_test_all = {}

    # Get the dataset mean and standard deviation
    mean_std_array = np.load("./dataset/federated_learning/mean_std.npy")
    mean = mean_std_array[0]
    std = mean_std_array[1]

    for device_type in device_types:
        # Compute dataset
        normal = []
        abnormal = []
        if mean_calc_needed:
            for file in os.listdir(f"./dataset/federated_learning/{device_type}/id_0{device_number}/normal"):
                if file != "mfcc":
                    normal.append(
                        np.load(f"./dataset/federated_learning/{device_type}/id_0{device_number}/normal/" + file,
                                allow_pickle=True).astype("float64"))

            for file in os.listdir(f"./dataset/federated_learning/{device_type}/id_0{device_number}/abnormal"):
                if file != "mfcc":
                    abnormal.append(
                        np.load(f"./dataset/federated_learning/{device_type}/id_0{device_number}/abnormal/" + file,
                                allow_pickle=True).astype("float64"))

            # Normalise data
            normal = (np.asarray(normal) - mean) / std
            abnormal = (np.asarray(abnormal) - mean) / std
        else:
            for file in os.listdir(f"./dataset_means_std/{device_type}/id_0{device_number}/normal"):
                if file != "mfcc":
                    normal.append(
                        np.load(f"./dataset_means_std/{device_type}/id_0{device_number}/normal/" + file,
                                allow_pickle=True).astype("float64"))

            for file in os.listdir(f"./dataset_means_std/{device_type}/id_0{device_number}/abnormal"):
                if file != "mfcc":
                    abnormal.append(
                        np.load(f"./dataset_means_std/{device_type}/id_0{device_number}/abnormal/" + file,
                                allow_pickle=True).astype("float64"))

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
                if mean_calc_needed:
                    means = []
                    stds = []
                    for filter in range(0, mfcc.shape[2]):
                        means.append(mfcc[channel, filter].mean())
                        stds.append(mfcc[channel, filter].std())

                    X_normal.append(tuple(means + stds))
                    Y_normal.append(class_labels["normal"])
                    Z_normal.append(counter)
                else:
                    X_normal.append(mfcc[channel])
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
                if mean_calc_needed:
                    means = []
                    stds = []
                    for filter in range(0, mfcc.shape[2]):
                        means.append(mfcc[channel, filter].mean())
                        stds.append(mfcc[channel, filter].std())

                    X_abnormal.append(tuple(means + stds))
                    Y_abnormal.append(class_labels["abnormal"])
                    Z_abnormal.append(counter)
                else:
                    X_abnormal.append(mfcc[channel])
                    Y_abnormal.append(class_labels["abnormal"])
                    Z_abnormal.append(counter)

            if VISUALISE:
                for i in range(0, mfcc.shape[0]):
                    plt.imshow(mfcc[i], interpolation="nearest", origin="lower", aspect="auto")
                    # plt.colorbar()
                    plt.show()

        # Set the abnormal variable to be garbage collected as it isn't needed anymore
        del abnormal

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


def get_val_and_training_set(X, Y):
    assert len(X.shape) == 2, "Give all device types in one matrix not one matrix for each type of device"

    # Split into train and test dataset
    randIndices = []
    random.seed(256)  # use the same seed everytime, so we always get the same results

    X_train = []
    Y_train = []
    X_val = []
    Y_val = []

    randIndices = []
    for i in range(0, int(7 * (len(X[:, 1].tolist()) / 10))):  # 70% of the data is used as training data
        randIndices.append(random.randint(0, len(X[:, 1].tolist()) - 1))

    indices = []
    for i in randIndices:
        X_train.append(X[i])
        Y_train.append(Y[i])

    indices = []
    for i in range(0, len(X[:, 1].tolist())):
        if i not in randIndices:
            X_val.append(X[i])
            Y_val.append(Y[i])

    return X_train, Y_train, X_val, Y_val


def find_best_sigma_for_C(X, Y, X_val, Y_val, C):
    for sigma in np.linspace(0, 10, 10):
        config = {
            "K": 2,
            "Ninit": 4,
            "PVinit": 16,
            "M": 1500,
            "C": C,
            "sigma": sigma,
            "threshold_type": 'fxd',
            "threshold_minimum": 0.05,
        }

        svm = LSSVM(config=config)
        svm.fit(X, Y)

        # variables for metrics calculated later
        right_number = 0

        for i in range(0, len(X_val)):
            prediction, score = svm.predict(tf.convert_to_tensor(X_val[i], dtype=tf.float64))
            if Y_val[i] == prediction:
                right_number += 1

        print(f"Accuracy for C = {C} and sigma = {sigma} is {str(round((right_number / len(X_val)) * 100, 6))}")


if __name__ == '__main__':
    tf.config.list_physical_devices("GPU")  # With this my home gpu is seen, if left out it is not
    init_gpu(devices="1", v=2)
    VISUALISE = False
    with_subsampling = False
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

    # # Load data
    # normal = []
    # abnormal = []
    # for device_type in device_types:
    #     for file in os.listdir(f"./dataset/federated_learning/{device_type}/id_00/normal"):
    #         try:
    #             if file != "mfcc":
    #                 normal.append(
    #                     np.load(f"./dataset/federated_learning/{device_type}/id_00/normal/" + file,
    #                             allow_pickle=True).astype("float32"))
    #         except:
    #             print(f"./dataset/federated_learning/{device_type}/id_00/normal/{file} not loaded")
    #
    #     for file in os.listdir(f"./dataset/federated_learning/{device_type}/id_00/abnormal"):
    #         try:
    #             if file != "mfcc":
    #                 abnormal.append(
    #                     np.load(f"./dataset/federated_learning/{device_type}/id_00/abnormal/" + file,
    #                             allow_pickle=True).astype("float32"))
    #         except:
    #             print(f"./dataset/federated_learning/{device_type}/id_00/abnormal/{file}")
    #
    # # Normalise data
    # mean_std_array = np.load("./dataset/federated_learning/mean_std.npy")
    # mean = mean_std_array[0]
    # std = mean_std_array[1]
    #
    # normal = (np.asarray(normal) - mean) / std
    # abnormal = (np.asarray(abnormal) - mean) / std
    #
    # # Dataset and labels
    # X_normal = []
    # Y_normal = []
    # X_abnormal = []
    # Y_abnormal = []
    # Z_normal = []
    # Z_abnormal = []
    #
    # counter = 0
    # subsampling_cnt = 0
    # for mfcc in normal:
    #     counter += 1
    #
    #     if with_subsampling:
    #         subsampling_cnt += 1
    #         if subsampling_cnt != 1:
    #             if subsampling_cnt >= 10:
    #                 subsampling_cnt = 0
    #             continue
    #
    #     for channel in range(0, mfcc.shape[0]):
    #         means = []
    #         stds = []
    #         for filter in range(0, mfcc.shape[2]):
    #             means.append(mfcc[channel, :, filter].mean())
    #             stds.append(mfcc[channel, :, filter].std())
    #
    #         X_normal.append(tuple(means + stds))
    #         Y_normal.append(class_labels["normal"])
    #         Z_normal.append(counter)
    #
    #     if VISUALISE:
    #         for i in range(0, mfcc.shape[0]):
    #             plt.imshow(mfcc[i], interpolation="nearest", origin="lower", aspect="auto")
    #             # plt.colorbar()
    #             plt.show()
    #
    # # Set the normal variable to be garbage collected as it isn't needed anymore
    # del normal
    #
    # # abnormal_rate = 1
    # # index = -1
    # subsampling_cnt = 0
    # for mfcc in abnormal:
    #     # index += 1
    #     # if index % abnormal_rate != 0 and abnormal_rate != 1:
    #     #     continue
    #
    #     counter += 1
    #
    #     if with_subsampling:
    #         subsampling_cnt += 1
    #         if subsampling_cnt != 1:
    #             if subsampling_cnt >= 10:
    #                 subsampling_cnt = 0
    #             continue
    #
    #     for channel in range(0, mfcc.shape[0]):
    #         means = []
    #         stds = []
    #         for filter in range(0, mfcc.shape[2]):
    #             means.append(mfcc[channel, :, filter].mean())
    #             stds.append(mfcc[channel, :, filter].std())
    #
    #     X_abnormal.append(tuple(means + stds))
    #     Y_abnormal.append(class_labels["abnormal"])
    #     Z_abnormal.append(counter)
    #
    #     if VISUALISE:
    #         for i in range(0, mfcc.shape[0]):
    #             plt.imshow(mfcc[i], interpolation="nearest", origin="lower", aspect="auto")
    #             # plt.colorbar()
    #             plt.show()
    #
    # # Set the abnormal variable to be garbage collected as it isn't needed anymore
    # del abnormal
    #
    # X = np.concatenate([X_normal, X_abnormal])
    # Y = np.concatenate([Y_normal, Y_abnormal])
    X_train_all, Y_train_all, X_test_all, Y_test_all = get_data_set(with_subsampling=with_subsampling)

    X_train, Y_train, X_val, Y_val = get_val_and_training_set(
        np.concatenate([X_train_all["fan"], X_train_all["pump"], X_train_all["valve"], X_train_all["slider"]]),
        np.concatenate([Y_train_all["fan"], Y_train_all["pump"], Y_train_all["valve"], Y_train_all["slider"]]))

    del X_train_all  # set these variables to be deleted form memory we don't need them anymore.
    del Y_train_all
    del X_test_all
    del Y_test_all

    # C_range = np.logspace(-2, 10, 13)  # taken from the examples of sklearn
    C_range = np.linspace(0, 0.1, 10)  # taken from the examples of sklearn

    processes = [Process(target=find_best_sigma_for_C, args=(X_train, Y_train, X_val, Y_val, C,)) for C in
                 C_range]  # Create processes to start all C searches in parallel

    for process in processes:  # Start each search process
        process.start()

    for process in processes:  # Wait till they are done
        process.join()

    print('\n\nDone', flush=True)

    # sigma_range = np.logspace(0, 6, 13)
    # param_grid = dict(C=C_range, sigma=sigma_range)
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=256)
    # grid = GridSearchCV(estimator=LSSVM(), scoring="accuracy", param_grid=param_grid, n_jobs=3, cv=cv)
    # grid.fit(X, Y)

    # print(
    #     "The best parameters are %s with a score of %0.2f"
    #     % (grid.best_params_, grid.best_score_)
    # )
