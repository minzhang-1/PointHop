import argparse
import pickle
import modelnet_data
import pointhop
import numpy as np
import data_utils
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--num_batch_train', type=int, default=20, help='Batch Number')
parser.add_argument('--num_batch_test', type=int, default=1, help='Batch Number')
parser.add_argument('--initial_point', type=int, default=1024, help='Point Number [256/512/1024/2048]')
parser.add_argument('--validation', default=False, help='Split train data or not')
parser.add_argument('--ensemble', default=False, help='Ensemble or not')
parser.add_argument('--rotation_angle', default=np.pi/4, help='Rotate angle')
parser.add_argument('--rotation_freq', default=8, help='Rotate time')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', default=[1024, 128, 128, 64], help='Point Number after down sampling')
parser.add_argument('--num_sample', default=[64, 64, 64, 64], help='KNN query number')
parser.add_argument('--num_filter', default=[15, 25, 40, 80], help='Filter Number ')
parser.add_argument('--pooling_method', default=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                                                 [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0],
                                                 [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1],
                                                 [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
                    help='Pooling methods [mean, max, l1, l2]')
FLAGS = parser.parse_args()

num_batch_train = FLAGS.num_batch_train
num_batch_test = FLAGS.num_batch_test
initial_point = FLAGS.initial_point
VALID = FLAGS.validation
ENSEMBLE = FLAGS.ensemble
angle_rotation = FLAGS.rotation_angle
freq_rotation = FLAGS.rotation_freq
num_point = FLAGS.num_point
num_sample = FLAGS.num_sample
num_filter = FLAGS.num_filter
pooling = FLAGS.pooling_method


LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def main():
    time_start = time.time()

    # load data
    train_data, train_label = modelnet_data.data_load(num_point=initial_point, data_dir=os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), train=True)
    test_data, test_label = modelnet_data.data_load(num_point=initial_point, data_dir=os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), train=False)

    # validation set
    if VALID:
        train_data, train_label, valid_data, valid_label = modelnet_data.data_separate(train_data, train_label)
    else:
        valid_data = test_data
        valid_label = test_label

    print(train_data.shape)
    print(valid_data.shape)

    if ENSEMBLE:
        angle = np.repeat(angle_rotation, freq_rotation)
    else:
        angle = [0]

    params = {}
    feat_train = []
    feat_valid = []
    for i in range(len(angle)):
        print('------------Train ', i, '--------------')
        idx_save, new_xyz_save, final_feature_train, feature_train, pca_params = \
            pointhop.pointhop_train(train_data, n_batch=num_batch_train, n_newpoint=num_point, n_sample=num_sample, layer_num=num_filter,
                                    energy_percent=None)
        print('------------Validation ', i, '--------------')

        final_feature_valid, feature_valid = pointhop.pointhop_pred(
            valid_data, n_batch=num_batch_test, pca_params=pca_params, n_newpoint=num_point, n_sample=num_sample, layer_num=num_filter,
            idx_save=None, new_xyz_save=None)

        feature_train = pointhop.extract(feature_train)
        feature_valid = pointhop.extract(feature_valid)
        feat_train.append(feature_train)
        feat_valid.append(feature_valid)
        params['stage %d pca_params' % i] = pca_params

        train_data = data_utils.data_augment(train_data, angle[i])
        valid_data = data_utils.data_augment(valid_data, angle[i])

    feat_train = np.concatenate(feat_train, axis=-1)
    feat_valid = np.concatenate(feat_valid, axis=-1)

    clf, acc_train, acc_valid, acc = pointhop.classify(feat_train, train_label, feat_valid, valid_label, pooling)
    params['clf'] = clf

    time_end = time.time()

    log_string("train acc is {}".format(acc_train))
    log_string('eval acc is {}'.format(acc_valid))
    log_string('eval mean acc is {}'.format(np.mean(acc)))
    log_string('per-class acc is {}'.format(str(acc)))
    log_string('totally time cost is {} minutes'.format((time_end - time_start)//60))

    with open(os.path.join(LOG_DIR, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)


if __name__ == '__main__':
    main()

