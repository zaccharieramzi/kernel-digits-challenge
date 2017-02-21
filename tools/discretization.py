import numpy as np


def discretize_orientation(patch_x, patch_y):

    a = np.arctan2(patch_y, patch_x)/(2*np.pi)
    a = round(a*8)/8*(2*np.pi)
    rounded_patch_x = np.cos(a)
    rounded_patch_y = np.sin(a)
    # round in 0-15
    discrete_or = round(np.arctan2(rounded_patch_y, rounded_patch_x)*8/np.pi)
    return discrete_or


def pin_as_vect(discrete_or):
    vect = np.zeros(16)
    for row in discrete_or:
        for el in row:
            vect[el] += 1
    return vect
