import argparse
import pickle
import sklearn
import modelnet_data
import pointhop
import numpy as np
import data_utils
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--num_batch_test', type=int, default=1, help='Batch Number')
parser.add_argument('--initial_point', type=int, default=1024, help='Point Number [256/512/1024/2048]')
parser.add_argument('--ensemble', default=False, help='Ensemble or not')
parser.add_argument('--rotation_angle', default=np.pi/4, help='Rotate angle')
parser.add_argument('--rotation_freq', default=8, help='Rotate time')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', default=[1024, 128, 128, 128], help='Point Number after down sampling')
parser.add_argument('--num_sample', default=[64, 64, 64, 64], help='KNN query number')
parser.add_argument('--num_filter', default=[15, 25, 40, 80], help='Filter Number ')
parser.add_argument('--pooling_method', default=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                                                 [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0],
                                                 [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1],
                                                 [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
                    help='Pooling methods [mean, max, l1, l2]')
FLAGS = parser.parse_args()

num_batch_test = FLAGS.num_batch_test
initial_point = FLAGS.initial_point
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
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def main():
    time_start = time.time()

    # load data
    test_data, test_label = modelnet_data.data_load(num_point=initial_point, data_dir='modelnet40_ply_hdf5_2048', train=False)

    if ENSEMBLE:
        angle = np.repeat(angle_rotation, freq_rotation)
    else:
        angle = [0]

    with open(os.path.join(LOG_DIR, 'params.pkl'), 'rb') as f:
        params = pickle.load(f, encoding='latin')

    # get feature and pca parameter
    feature_test = []
    for i in range(len(angle)):
        print('------------Test ', i, '--------------')

        pca_params = params['stage %d pca_params' % i]

        final_feature, feature = pointhop.pointhop_pred(
            test_data, n_batch=num_batch_test, pca_params=pca_params, n_newpoint=num_point, n_sample=num_sample, layer_num=num_filter,
            idx_save=None, new_xyz_save=None)

        feature = pointhop.extract(feature)

        feature_test.append(feature)
        test_data = data_utils.data_augment(test_data, angle[i])

    feature_test = np.concatenate(feature_test, axis=-1)

    clf_tmp = params['clf']
    pred_test_tmp = []
    acc_test_tmp = []
    for i in range(len(pooling)):
        clf = clf_tmp['pooling method %d' % i]
        feature_test_tmp = pointhop.aggregate(feature_test, pooling[i])
        pred_test = clf.predict(feature_test_tmp)
        acc_test = sklearn.metrics.accuracy_score(test_label, pred_test)
        pred_test_tmp.append(pred_test)
        acc_test_tmp.append(acc_test)
    idx = np.argmax(acc_test_tmp)
    pred_test = pred_test_tmp[idx]
    acc = pointhop.average_acc(test_label, pred_test)

    time_end = time.time()

    log_string("test acc is {}".format(acc_test_tmp[idx]))
    log_string('test mean acc is {}'.format(np.mean(acc)))
    log_string('per-class acc is {}'.format(str(acc)))
    log_string('totally time cost is {} minutes'.format((time_end - time_start)//60))

    # with open(os.path.join(LOG_DIR, 'ensemble_pred_test.pkl'), 'wb') as f:
    #     pickle.dump(pred_test, f)


if __name__ == '__main__':
    main()

