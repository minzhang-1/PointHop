import os
import h5py
import numpy as np
from sklearn.model_selection import train_test_split


def load_dir(data_dir, name='train_files.txt'):
    with open(os.path.join(data_dir,name),'r') as f:
        lines = f.readlines()
    return [os.path.join(data_dir, line.rstrip().split('/')[-1]) for line in lines]


def shuffle_data(data):
    """ Shuffle data order.
        Input:
          data: B,N,... numpy array
        Return:
          shuffled data, shuffle indices
    """
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    return data[idx, ...], idx


def shuffle_points(data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(data.shape[1])
    np.random.shuffle(idx)
    return data[:, idx, :], idx


def xyz2sphere(data):
    """
    Input: data(B,N,3) xyz_coordinates
    Return: data(B,N,3) sphere_coordinates
    """
    r = np.sqrt(np.sum(data**2, axis=2, keepdims=False))
    theta = np.arccos(data[...,2]*1.0/r)
    phi = np.arctan(data[...,1]*1.0/data[...,0])

    if len(r.shape) == 2:
        r = np.expand_dims(r, 2)
    if len(theta.shape) == 2:
        theta = np.expand_dims(theta, 2)
    if len(phi.shape) == 2:
        phi = np.expand_dims(phi, 2)

    data_sphere = np.concatenate([r, theta, phi], axis=2)
    return data_sphere


def xyz2cylind(data):
    """
    Input: data(B,N,3) xyz_coordinates
    Return: data(B,N,3) cylindrical_coordinates
    """
    r = np.sqrt(np.sum(data[...,:2]**2, axis=2, keepdims=False))
    phi = np.arctan(data[...,1]*1.0/data[...,0])
    z = data[...,2]

    if len(r.shape) == 2:
        r = np.expand_dims(r, 2)
    if len(z.shape) == 2:
        z = np.expand_dims(z, 2)
    if len(phi.shape) == 2:
        phi = np.expand_dims(phi, 2)

    data_sphere = np.concatenate([r, z, phi], axis=2)
    return data_sphere


def data_load(num_point=None, data_dir='/modelnet40_ply_hdf5_2048', train=True):
    if not os.path.exists('modelnet40_ply_hdf5_2048'):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('rm %s' % (zipfile))

    if train:
        data_pth = load_dir(data_dir, name='train_files.txt')
    else:
        data_pth = load_dir(data_dir, name='test_files.txt')

    point_list = []
    label_list = []
    for pth in data_pth:
        data_file = h5py.File(pth, 'r')
        point = data_file['data'][:]
        label = data_file['label'][:]
        point_list.append(point)
        label_list.append(label)
    data = np.concatenate(point_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    # data, idx = shuffle_data(data)
    # data, ind = shuffle_points(data)

    if not num_point:
        return data[:, :, :], label
    else:
        return data[:, :num_point, :], label


def data_separate(data, label):
    seed = 7
    np.random.seed(seed)
    train_data, valid_data, train_label, valid_label = train_test_split(data, label, test_size=0.1, random_state=seed)

    return train_data, train_label, valid_data, valid_label


