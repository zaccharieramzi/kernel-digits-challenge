from math import sqrt
import os

import numpy as np
import PIL.Image as Img

from .data_loading import load_labels, load_images


def dump_as_png(type='test', number=None):
    '''
    Dumps images in different folder for each label
    Args :
           - type (str): "train" or "test"
           - number (int): number of dumped images
    '''
    if type not in ["train", "test"]:
        raise ValueError("type {} unknown, it should be either"
                         "'train' or 'test'".format(type))

    X = load_images(type=type)
    # for training images we also load the labels for naming puproses
    if type == "train":
        Y = load_labels()
    # randomly sampling in images
    if number is not None:
        indices = np.random.choice(X.shape[0], number)
        X = X[indices]
        if type == "train":
            Y = Y[indices]

    X = reshape_as_images(X)

    # rescaling
    X = X - X.min(axis=(1, 2))[:, None, None, :]
    X = X / X.max(axis=(1, 2))[:, None, None, :]

    # format as 8 bit
    X *= 255
    X = X.astype(np.uint8)

    # we put results in a directory
    dir_path = type + "_images"
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    if type == 'train':
        # we create a directory per label
        subdirs = []
        for i in range(int(Y.max()+1)):
            subdir = os.path.join(dir_path, str(i))
            subdirs.append(subdir)
            if not os.path.isdir(subdir):
                os.mkdir(subdir)
    for i in range(X.shape[0]):
        if type == "train":
            filename = os.path.join(subdirs[Y[i]], "{:d}.png".format(i))
        else:
            filename = os.path.join(dir_path, "{:d}.png".format(i))
        Img.fromarray(X[i]).save(filename)


def reshape_as_images(X):
    '''
    Reshape the data as a vector of images
    '''
    n_images, img_len = X.shape
    img_size = int(sqrt(img_len // 3))

    imgs = np.empty((n_images, img_size, img_size, 3), dtype=X.dtype)
    for c in range(3):
        imgs[:, :, :, c] = X[:, 1024 * c:1024 * (c + 1)].reshape(
            (n_images, img_size, img_size))
    return imgs
