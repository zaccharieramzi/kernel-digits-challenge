import numpy as np

from tools.quantization import vf_vector


def process_images(n, centroids, pins, pin_to_im):
    '''Vectorize the features of each images contained in the rows of X.
    '''
    # X_proc = np.zeros((n, centroids.shape[0]))
    X_proc = vf_vector(pins, centroids, pin_to_im, n)
    return X_proc
