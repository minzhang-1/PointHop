import numpy as np
import sklearn
from sklearn.decomposition import PCA, IncrementalPCA
from numpy import linalg as LA

from sklearn import svm
from sklearn import ensemble


import point_utils


def pointhop_train(train_data, n_batch, n_newpoint, n_sample, layer_num, energy_percent):
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
    batch_size = num_data//n_batch
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

        print('Start query and gathering-------------')
        # time_start = time.time()
        if not grouped_feature is None:
            idx, grouped_feature = query_and_gather(new_xyz, n_batch, batch_size, point_data, grouped_feature, n_sample[i], None)
        else:
            idx, grouped_feature = query_and_gather(new_xyz, n_batch, batch_size, point_data, feature_data, n_sample[i], None)

        idx_save['Layer_%d' % (i)] = idx
        grouped_feature = grouped_feature.reshape(num_data*n_newpoint[i], -1)
        print('ok-------------')

        kernels, mean = find_kernels_pca(grouped_feature, layer_num[i], energy_percent, n_batch)
        if i == 0:
            transformed = np.matmul(grouped_feature, np.transpose(kernels))
        else:
            bias = LA.norm(grouped_feature, axis=1)
            bias = np.max(bias)
            pca_params['Layer_{:d}/bias'.format(i)] = bias
            grouped_feature = grouped_feature + bias

            transformed = np.matmul(grouped_feature, np.transpose(kernels))
            e = np.zeros((1, kernels.shape[0]))
            e[0, 0] = 1
            transformed -= bias*e
        grouped_feature = transformed.reshape(num_data, n_newpoint[i], -1)
        print(grouped_feature.shape)
        feature_train.append(grouped_feature)
        pca_params['Layer_{:d}/kernel'.format(i)] = kernels
        pca_params['Layer_{:d}/pca_mean'.format(i)] = mean
        point_data = new_xyz
    final_feature = grouped_feature.max(axis=1, keepdims=False)

    return idx_save, new_xyz_save, final_feature, feature_train, pca_params


def pointhop_pred(test_data, n_batch, pca_params, n_newpoint, n_sample, layer_num, idx_save, new_xyz_save):
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
    batch_size = num_data//n_batch

    feature_data = test_data

    for i in range(len(n_newpoint)):
        if not new_xyz_save:
            point_num = point_data.shape[1]
            if n_newpoint[i] == point_num:
                new_xyz = point_data
            else:
                new_xyz = point_utils.furthest_point_sample(point_data, n_newpoint[i])
        else:
            print('---------------loading idx--------------')
            new_xyz = new_xyz_save['Layer_{:d}'.format(i)]

        if not grouped_feature is None:
            idx, grouped_feature = query_and_gather(new_xyz, n_batch, batch_size, point_data, grouped_feature, n_sample[i], None)
        else:
            idx, grouped_feature = query_and_gather(new_xyz, n_batch, batch_size, point_data, feature_data, n_sample[i], None)

        grouped_feature = grouped_feature.reshape(num_data*n_newpoint[i], -1)

        kernels = pca_params['Layer_{:d}/kernel'.format(i)]
        mean = pca_params['Layer_{:d}/pca_mean'.format(i)]

        if i == 0:
            transformed = np.matmul(grouped_feature, np.transpose(kernels))
        else:
            bias = pca_params['Layer_{:d}/bias'.format(i)]
            grouped_feature = grouped_feature + bias
            transformed = np.matmul(grouped_feature, np.transpose(kernels))
            e = np.zeros((1, kernels.shape[0]))
            e[0, 0] = 1
            transformed -= bias*e
        grouped_feature = transformed.reshape(num_data, n_newpoint[i], -1)
        feature_test.append(grouped_feature)
        point_data = new_xyz
    final_feature = grouped_feature.max(axis=1, keepdims=False)
    return final_feature, feature_test


def query_and_gather(new_xyz, n_batch, batch_size, pts_coor, pts_fea, n_sample, pooling):
    idx = []
    grouped_feature = []
    for j in range(n_batch):
        if j != n_batch - 1:
            idx_tmp = point_utils.knn(new_xyz[j * batch_size:(j + 1) * batch_size],
                                      pts_coor[j * batch_size:(j + 1) * batch_size]
                                      , n_sample)
            grouped_feature_tmp = point_utils.gather_fea(idx_tmp, pts_coor[j * batch_size:(j + 1) * batch_size],
                                                         pts_fea[j * batch_size:(j + 1) * batch_size])
        else:
            idx_tmp = point_utils.knn(new_xyz[j * batch_size:], pts_coor[j * batch_size:], n_sample)
            grouped_feature_tmp = point_utils.gather_fea(idx_tmp, pts_coor[j * batch_size:],
                                                         pts_fea[j * batch_size:])
        if pooling is not None:
            grouped_feature_tmp = grouped_feature_tmp.reshape(grouped_feature_tmp.shape[0], grouped_feature_tmp.shape[1], 8, -1)
            grouped_feature_tmp = extract(grouped_feature_tmp, pooling, 2)
        idx.append(idx_tmp)
        grouped_feature.append(grouped_feature_tmp)
    idx = np.concatenate(idx, axis=0)
    grouped_feature = np.concatenate(grouped_feature, axis=0)
    return idx, grouped_feature


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


def find_kernels_pca(sample_patches, num_kernels, energy_percent, n_batch):
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

    # pca = PCA(n_components=training_data.shape[1], svd_solver='full', whiten=True)
    batch_size = training_data.shape[0]//n_batch
    pca = IncrementalPCA(n_components=training_data.shape[1], whiten=True, batch_size=batch_size, copy=False)
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

    feature = {}
    label = {}
    feature['train'] = feat_tmp_train
    feature['test'] = feat_tmp_valid
    label['train'] = train_label
    label['test'] = valid_label
    import os
    import pickle
    with open(os.path.join('/home/minzhang/pointhop-master/feat.pkl'), 'wb') as f:
        pickle.dump(feature, f)
    with open(os.path.join('/home/minzhang/pointhop-master/label.pkl'), 'wb') as f:
        pickle.dump(label, f)
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








