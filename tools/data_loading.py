import os

import numpy as np
import h5py
import PIL.Image as Im


def load_labels():
    '''
    load labels as a nd array of unsigned int of 8 bits
    '''
    Y = np.loadtxt("data/Ytr.csv",
                   skiprows=1, usecols=(1,), delimiter=',').astype('uint8')
    return Y


def dummy_code(Y):
    '''
    Args :
           - type nd array  : Labels as a vector of float
    Returns :
           - ndarray (., n_classes) labels dummy coded
    '''
    n = Y.shape[0]
    Y_dum = np.zeros((n, int(Y.max()) + 1))
    Y_dum[np.arange(n), Y.astype('uint8')] = 1
    return Y_dum


def load_images(type="train"):
    '''
    Args :
           - type (str): "train" or "test"
    Returns :
           - X ndarray (5000,3072)
    '''
    if type == "train":
        path = 'data/Xtr.csv'
        image_size = 3072  # 32*32*3
    elif type == "test":
        path = 'data/Xte.csv'
        image_size = 3072
    elif type == "train63":
        path = 'data/Xtr63.csv'
        image_size = 11907  # 63*63*3
    elif type == "test63":
        path = 'data/Xte63.csv'
        image_size = 11907
    else:
        raise ValueError("type {} unknown, it should be either"
                         "'train' or 'test'".format(type))

    with open(path) as f:
        lines = f.readlines()
        X = np.empty((len(lines), 3072), dtype=np.float32)

        for i, line in enumerate(lines):
            X[i] = np.fromstring(line[:line.index(',\n')], dtype=np.float32,
                                 sep=",")
        return X


def load_images_resized(type="train"):
    '''
    Args :
           - type (str): "train" or "test"
    Returns :
           - X ndarray (5000,3072)
    '''
    if type not in ("train", "test"):
        raise ValueError("type {} unknown, it should be either"
                         "'train' or 'test'".format(type))
    dataset_path = "data/{:s}_resized.hdf5".format(type)

    with h5py.File(dataset_path, "r") as h5f:
        return np.array(h5f["images"])


def store_resized_images(type="train"):
    if type == "train":
        root_path = "data/images_train_resize"
    elif type == "test":
        root_path = "data/test_images_resize"
    else:
        raise ValueError("type {} unknown, it should be either"
                         "'train' or 'test'".format(type))

    if not os.path.isdir(root_path):
        print("Data not at expected path : '{:s}'".format(root_path))

    if type == "train":
        images = np.empty((5000, 63, 63))
        for i in range(10):
            print("folder number {}".format(i))
            im_dir = os.path.join(root_path, str(i))

            images_path = [os.path.join(im_dir, f) for f in os.listdir(im_dir)
                           if f[-4:] == ".png"]
            for im in images_path:
                index = int(im.split('.')[0].split('/')[-1])
                images[index] = np.array(Im.open(im))

    else:
        im_dir = root_path
        images_path = [os.path.join(im_dir, f) for f in os.listdir(im_dir)
                       if f[-4:] == ".png"]

        images = np.empty((len(images_path), 63, 63))
        for im in images_path:
            index = int(im.split('.')[0].split('/')[-1])
            images[index] = np.array(Im.open(im))

    with h5py.File(os.path.join("data", type + "_resized.hdf5"), "w") as h5f:
        dset = h5f.create_dataset("images", data=images)
        h5f.flush()
