import numpy as np
import time

from .corner_response_function import convolve, difference_of_Gaussian_filters\
    compute_corner_response
from .data_loading import load_images
from .discretization import discretize_orientation, pin_as_vect
from .visualization import reshape_as_images, imshow


def pins_generation(training_idx=[], window_size=5, stride=3, patch_size=5,
                    filter_size=3, filter_sigma=0.25,
                    ratio_pins_per_image=25, data_type="train",
                    index_to_visualize=[]):
    # data loading
    X = load_images(type=data_type)
    image_list = reshape_as_images(X)
    image_list = image_list.mean(axis=3)
    n_images = image_list.shape[0]

    # R contains all the corner response functions for all images.
    # Each corner response function is given by a matrix of size
    # R_size x R_size, which gives for each window running through the image
    # the corner response function.
    R_size = (image_list.shape[1] - window_size - filter_size)//stride + 1
    R = np.zeros((n_images, R_size, R_size))

    im_size = image_list[0].shape[0]  # normally 32
    filterx, filtery = difference_of_Gaussian_filters(shape=(5, 5),
                                                      sigma=filter_sigma)

    # In this loop, for each image, we compute the corner response function.
    for image_index in range(n_images):
        image_mat = image_list[image_index]
        image_grad_x, image_grad_y = convolve(image_mat, filterx, filtery)

        for i in range(R_size):
            for j in range(R_size):
                I_x = image_grad_x[i*stride:i*stride+window_size,
                                   j*stride:j*stride+window_size]
                I_y = image_grad_y[i*stride:i*stride+window_size,
                                   j*stride:j*stride+window_size]

                R[image_index, i, j] = compute_corner_response(I_x, I_y)
        # absoluting of R (because we are interested in the absolute value)
        R[image_index] = np.abs(R[image_index])
        if image_index in index_to_visualize:
            i, j = np.where(
                R[image_index] > np.percentile(R[image_index],
                                               100 - ratio_pins_per_image))
            i, j = from_R_to_im(x, y, window_size, stride, filter_size)

            x, y = j, i

            heatmap = R_to_heatmap(R[image_index], window_size, stride,
                                   im_size)
            imshow(image_list[image_index], points_of_interest=(x, y),
                   heatmap=heatmap)
            continue

    pins = list()
    train_pins = list()
    pin_to_im = dict()
    # In this loop, we retrieve the gradient patch associated with each point
    # that we identified as interesting. We then discretize and vectorize it.
    for image_idx in range(n_images):
        image_mat = image_list[image_idx]
        i_s, j_s = np.where(
            R[image_idx] > np.percentile(R[image_idx],
                                         100 - ratio_pins_per_image))
        i_s, j_s = from_R_to_im(i_s, j_s, window_size, stride, filter_size)
        image_grad_x, image_grad_y = compute_gaussian_grad(image_mat)
        for i, j in zip(i_s, j_s):  # i, j are the coordinates in R of POI
            patch_x = image_grad_x[i-patch_size//2:i+patch_size//2+1,
                                   j-patch_size//2:j+patch_size//2+1]
            patch_y = image_grad_y[i-patch_size//2:i+patch_size//2+1,
                                   j-patch_size//2:j+patch_size//2+1]
            pin_as_matrix = discretize_orientation(patch_x, patch_y)
            pin = pin_as_vect(pin_as_matrix)
            pins.append(pin)  # pins is list of all pins for all images
            # for visualization purposes
            pin_to_im[len(pins)-1] = image_idx
            if image_idx in training_idx:
                train_pins.append(pin)

    return {
        "pins": pins,
        "train_pins": train_pins,
        "pin_to_im": pin_to_im
    }


def from_R_to_im(x, y, window_size, stride, filter_size):
    x = x * stride + (window_size - 1)//2 + (filter_size - 1)//2
    y = y * stride + (window_size - 1)//2 + (filter_size - 1)//2

    return (x, y)


def R_to_heatmap(R, window_size, stride, im_size):
    R_size = R.shape[0]
    # create heatmap
    heatmap = np.zeros((im_size, im_size))
    for i in range(R_size):
        for j in range(R_size):
            ix, iy = from_R_to_im(i, j, window_size, stride)
            heatmap[ix:ix+window_size,
                    iy:iy+window_size] = R[i, j]
    heatmap /= heatmap.max()
    return heatmap
