import numpy as np
import sklearn
from sklearn.decomposition import PCA
from numpy import linalg as LA

from sklearn import svm
from sklearn import ensemble


import point_utils
import modelnet_data

import threading


def pointhop_train(train_data, n_newpoint, n_sample, layer_num, energy_percent):
    '''
    Train based on the provided samples.
    :param train_data: [num_samples, num_point, feature_dimension]
    :param n_newpoint: point numbers used in every stage
    :param n_sample: k nearest neighbors
    :param layer_num: num kernels to be preserved
    :param energy_percent: the percent of energy to be preserved
    :return: idx, new_idx, final stage feature, feature, pca_params
    '''

    num_data = train_data.shape[0]
    pca_params = {}
    idx_save = {}
    new_xyz_save = {}

    point_data = train_data
    grouped_feature = None
    feature_train = []

    feature_data = train_data

    for i in range(len(n_newpoint)):
        print(i)
        point_num = point_data.shape[1]
        print('Start sampling-------------')
        if n_newpoint[i] == point_num:
            new_xyz = point_data
        else:
            new_xyz = point_utils.furthest_point_sample(point_data, n_newpoint[i])

        new_xyz_save['Layer_{:d}'.format(i)] = new_xyz

        print('Start query-------------')
        for k in range(len(n_sample[i])):
            idx = point_utils.knn(new_xyz, point_data, n_sample[i][k])
            idx_save['Layer_%d_%d' % (i, k)] = idx

            print('Start Gathering-------------')
            # time_start = time.time()
            if not grouped_feature is None:
                grouped_feature_temp = point_utils.gather_fea(idx, point_data, grouped_feature, n_sample[i][k])
            else:
                grouped_feature_temp = point_utils.gather_fea(idx, point_data, feature_data, n_sample[i][k])
            grouped_feature_temp = grouped_feature_temp.reshape(num_data*n_newpoint[i], -1)
            print('ok-------------')
            for j in range(len(layer_num[i])):
                kernels, mean = find_kernels_pca(grouped_feature_temp, layer_num[i][j], energy_percent)
                if i == 0 and j == 0:
                    transformed = np.matmul(grouped_feature_temp, np.transpose(kernels))
                else:
                    bias = LA.norm(grouped_feature_temp, axis = 1)
                    bias = np.max(bias)
                    pca_params['Layer_{:d}_{:d}_{:d}/bias'.format(i,j,k)] = bias
                    # grouped_feature_centered_w_bias = grouped_feature + np.sqrt(layer_num[i][j])*bias
                    grouped_feature_centered_w_bias = grouped_feature_temp + bias

                    transformed = np.matmul(grouped_feature_centered_w_bias, np.transpose(kernels))
                    e = np.zeros((1, kernels.shape[0]))
                    e[0, 0] = 1
                    transformed -= bias*e
                feature_train.append((transformed.reshape(num_data, n_newpoint[i], -1)))
                pca_params['Layer_{:d}_{:d}_{:d}/kernel'.format(i, j, k)] = kernels
                pca_params['Layer_{:d}_{:d}_{:d}/pca_mean'.format(i, j, k)] = mean

                if k == 0:
                    grouped_feature_temp2 = transformed
                else:
                    grouped_feature_temp2 = np.concatenate((grouped_feature_temp2, transformed), axis=-1)
        grouped_feature = grouped_feature_temp2
        grouped_feature = grouped_feature.reshape(num_data, n_newpoint[i], -1)
        print(grouped_feature.shape)
        point_data = new_xyz

    final_feature = grouped_feature.max(axis=1, keepdims=False)

    return idx_save, new_xyz_save, final_feature, feature_train, pca_params


def pointhop_pred(test_data, pca_params, n_newpoint, n_sample, layer_num, idx_save, new_xyz_save):
    '''
    Test based on the provided samples.
    :param test_data: [num_samples, num_point, feature_dimension]
    :param pca_params: pca kernel and mean
    :param n_newpoint: point numbers used in every stage
    :param n_sample: k nearest neighbors
    :param layer_num: num kernels to be preserved
    :param idx_save: knn index
    :param new_xyz_save: down sample index
    :return: final stage feature, feature, pca_params
    '''

    num_data = test_data.shape[0]
    point_data = test_data
    grouped_feature = None
    feature_test = []

    feature_data = test_data

    for i in range(len(n_newpoint)):
        if not idx_save:
            point_num = point_data.shape[1]
            if n_newpoint[i] == point_num:
                new_xyz = point_data
            else:
                new_xyz = point_utils.furthest_point_sample(point_data, n_newpoint[i])
        else:
            print('---------------loading idx--------------')
            new_xyz = new_xyz_save['Layer_{:d}'.format(i)]

        for k in range(len(n_sample[i])):
            if idx_save:
                idx = idx_save['Layer_%d_%d' % (i, k)]
            else:
                idx = point_utils.knn(new_xyz, point_data, n_sample[i][k])

            if not grouped_feature is None:
                grouped_feature_temp = point_utils.gather_fea(idx, point_data, grouped_feature, n_sample[i][k])
            else:
                grouped_feature_temp = point_utils.gather_fea(idx, point_data, feature_data, n_sample[i][k])

            grouped_feature_temp = grouped_feature_temp.reshape(num_data*n_newpoint[i], -1)

            for j in range(len(layer_num[i])):
                kernels = pca_params['Layer_{:d}_{:d}_{:d}/kernel'.format(i, j, k)]
                mean = pca_params['Layer_{:d}_{:d}_{:d}/pca_mean'.format(i, j, k)]

                if i == 0 and j == 0:
                    transformed = np.matmul(grouped_feature_temp, np.transpose(kernels))
                else:
                    bias = pca_params['Layer_{:d}_{:d}_{:d}/bias'.format(i, j, k)]
                    grouped_feature_centered_w_bias = grouped_feature_temp + bias
                    transformed = np.matmul(grouped_feature_centered_w_bias, np.transpose(kernels))
                    e = np.zeros((1, kernels.shape[0]))
                    e[0, 0] = 1
                    transformed -= bias*e
                feature_test.append((transformed.reshape(num_data, n_newpoint[i], -1)))
                if k == 0:
                    grouped_feature_temp2 = transformed
                else:
                    grouped_feature_temp2 = np.concatenate((grouped_feature_temp2, transformed), axis=-1)
        grouped_feature = grouped_feature_temp2
        grouped_feature = grouped_feature.reshape(num_data, n_newpoint[i], -1)
        point_data = new_xyz

    final_feature = grouped_feature.max(axis=1, keepdims=False)
    return final_feature, feature_test


def remove_mean(features, axis):
    '''
    Remove the dataset mean.
    :param features [num_samples,...]
    :param axis the axis to compute mean
    
    '''
    feature_mean = np.mean(features, axis=axis, keepdims=True)
    feature_remove_mean = features-feature_mean
    return feature_remove_mean, feature_mean


def remove_zero_patch(samples):
    std_var = (np.std(samples, axis=1)).reshape(-1, 1)
    ind_bool = (std_var == 0)
    ind = np.where(ind_bool==True)[0]
    # print('zero patch shape:',ind.shape)
    samples_new = np.delete(samples, ind, 0)
    return samples_new


def find_kernels_pca(sample_patches, num_kernels, energy_percent):
    '''
    Do the PCA based on the provided samples.
    If num_kernels is not set, will use energy_percent.
    If neither is set, will preserve all kernels.
    :param samples: [num_samples, feature_dimension]
    :param num_kernels: num kernels to be preserved
    :param energy_percent: the percent of energy to be preserved
    :return: kernels, sample_mean
    '''
    # Remove patch mean
    sample_patches_centered, dc = remove_mean(sample_patches, axis=1)
    sample_patches_centered = remove_zero_patch(sample_patches_centered)
    # Remove feature mean (Set E(X)=0 for each dimension)
    training_data, feature_expectation = remove_mean(sample_patches_centered, axis=0)

    pca = PCA(n_components=training_data.shape[1], svd_solver='full', whiten=True)
    pca.fit(training_data)

    # Compute the number of kernels corresponding to preserved energy
    if energy_percent:
        energy = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.sum(energy < energy_percent)+1
    else:
        num_components = num_kernels

    kernels = pca.components_[:num_components, :]
    mean = pca.mean_

    num_channels = sample_patches.shape[-1]
    largest_ev = [np.var(dc*np.sqrt(num_channels))]
    dc_kernel = 1/np.sqrt(num_channels)*np.ones((1, num_channels))/np.sqrt(largest_ev)
    kernels = np.concatenate((dc_kernel, kernels), axis=0)

    print("Num of kernels: %d" % num_components)
    print("Energy percent: %f" % np.cumsum(pca.explained_variance_ratio_)[num_components-1])
    return kernels, mean


def extract(feat):
    '''
    Do feature extraction based on the provided feature.
    :param feat: [num_layer, num_samples, feature_dimension]
    # :param pooling: pooling method to be used
    :return: feature
    '''
    mean = []
    maxi = []
    l1 = []
    l2 = []

    for i in range(len(feat)):
        mean.append(feat[i].mean(axis=1, keepdims=False))
        maxi.append(feat[i].max(axis=1, keepdims=False))
        l1.append(np.linalg.norm(feat[i], ord=1, axis=1, keepdims=False))
        l2.append(np.linalg.norm(feat[i], ord=2, axis=1, keepdims=False))
    mean = np.concatenate(mean, axis=-1)
    maxi = np.concatenate(maxi, axis=-1)
    l1 = np.concatenate(l1, axis=-1)
    l2 = np.concatenate(l2, axis=-1)

    return [mean, maxi, l1, l2]


def aggregate(feat, pool):
    feature = []
    for j in range(len(feat)):
        feature.append(feat[j] * pool[j])
    feature = np.concatenate(feature, axis=-1)
    return feature


def classify(feature_train, train_label, feature_valid, valid_label, pooling):
    '''
    Train classifier based on the provided feature.
    :param feature_train: [num_samples, feature_dimension]
    :param train_label: train label provided
    :param feature_valid: [num_samples, feature_dimension]
    :param valid_label: train label provided
    :param pooling: pooling methods provided
    :return: classifer, train accuracy, evaluate accuracy
    '''

    clf_tmp = {}
    acc_train = []
    acc_valid = []
    pred_valid = []
    for i in range(len(pooling)):
        feat_tmp_train = aggregate(feature_train, pooling[i])
        feat_tmp_valid = aggregate(feature_valid, pooling[i])
        clf = rf_classifier(feat_tmp_train, np.squeeze(train_label))
        pred_train = clf.predict(feat_tmp_train)
        acc_train.append(sklearn.metrics.accuracy_score(train_label, pred_train))
        pred_valid_tmp = clf.predict(feat_tmp_valid)
        pred_valid.append(pred_valid_tmp)
        acc_valid.append(sklearn.metrics.accuracy_score(valid_label, pred_valid_tmp))
        clf_tmp['pooling method %d' % i] = clf
    idx = np.argmax(acc_valid)
    acc = average_acc(valid_label, pred_valid[idx])
    # print(pooling[idx])

    return clf_tmp, acc_train[idx], acc_valid[idx], acc


def average_acc(label, pred_label):

    classes = np.arange(40)
    acc = np.zeros(len(classes))
    for i in range(len(classes)):
        ind = np.where(label == classes[i])[0]
        pred_test_special = pred_label[ind]
        acc[i] = len(np.where(pred_test_special == classes[i])[0])/float(len(ind))
    return acc


def onehot_encoding(n_class, labels):

    targets = labels.reshape(-1)
    one_hot_targets = np.eye(n_class)[targets]
    return one_hot_targets


# SVM
def svm_classifier(feat, y):
    '''
    Train svm based on the provided feature.
    :param feat: [num_samples, feature_dimension]
    :param y: label provided
    :return: classifer
    '''
    clf = svm.SVC(probability=True,gamma='auto')
    clf.fit(feat, y)
    return clf


# RF
def rf_classifier(feat, y):
    '''
    Train svm based on the provided feature.
    :param feat: [num_samples, feature_dimension]
    :param y: label provided
    :return: classifer
    '''
    clf = ensemble.RandomForestClassifier(n_estimators=128, bootstrap=False,
                                          n_jobs=-1)
    clf.fit(feat, y)
    return clf








