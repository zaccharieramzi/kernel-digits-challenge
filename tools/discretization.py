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
    a = np.arctan2(patch_y, patch_x)/(2*np.pi)
    a = np.round(a*8)/8*(2*np.pi)
    rounded_patch_x = np.cos(a)
    rounded_patch_y = np.sin(a)
    # round in 0-15
    discrete_or = np.round(np.arctan2(rounded_patch_y,
                                      rounded_patch_x)*8/np.pi)
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
            vect[el] += 1
    return vect
