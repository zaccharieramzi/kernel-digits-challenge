import numpy as np


def load_labels():
    '''
    load labels as a nd array ou unsigned int of 8 bits
    '''
    Y = np.loadtxt("../data/Ytr.csv",
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
    Y_dum = np.zeros((n, int(Y.max())+1))
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
        path = '../data/Xtr.csv'
    elif type == "test":
        path = '../data/Xte.csv'
    else:
        print("Type Error : argument should be either 'train' or 'test'")
        return

    f = open(path)
    lines = f.readlines()
    X = np.empty((len(lines), 3072), dtype=np.float32)

    for i, line in enumerate(lines):
        X[i] = np.fromstring(line[:line.index(',\n')], dtype=np.float32,
                             sep=",")
    return X
