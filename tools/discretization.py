import numpy as np


def discretize_orientation(patch_x, patch_y):
    '''
    args :
        - patch_x: gradient in x of the patch in (x, y)
        - patch_y: gradient in y of the patch in (x, y)
    return
        - discrete_or: an array containing the orientation discretized between
        0 and 15*pi/8
    '''
    orientation = np.arctan2(patch_y, patch_x)
    discrete_or = np.round(8*orientation / np.pi)
    return discrete_or


def pin_as_vect(discrete_or):
    '''
    Converting the orientation matrix to a pin (to be used in kmeans)
    args :
        - discrete_or: pin as a matrix patch_size*patch_size
    return
        - vect: pin as a vector counting orientation occurences
    '''
    vect = np.zeros(16)
    for row in discrete_or:
        for el in row:
            if el >= 0:
                idx = int(el)
            else:
                idx = 16 + int(el)
            vect[idx] += 1
    return vect
