import numpy as np
import time

from .corner_response_function import compute_gaussian_grad,\
                                                compute_corner_respons
from .visualization import reshape_as_images, imshow
from .data_loading import load_images

FILTER_SIZE = 3


def feature_learning():
    X = load_images(type="train")
    image_list = reshape_as_images(X)
    image_list = image_list.mean(axis=3)
    window_size = 5
    stride = 3
    patch_size = 5
    pins = list()

    R_size = (image_list.shape[1] - window_size - FILTER_SIZE)//stride + 1
    R = np.zeros((image_list.shape[0], R_size, R_size))

    for ind in range(image_list.shape[0]):
        image_mat = image_list[ind]
        im_size = image_mat.shape[0]  # normally 32
        image_grad_x, image_grad_y = compute_gaussian_grad(image_mat)

        for i in range(R_size):
            for j in range(R_size):
                I_x = image_grad_x[i*stride:i*stride+window_size,
                                   j*stride:j*stride+window_size]
                I_y = image_grad_y[i*stride:i*stride+window_size,
                                   j*stride:j*stride+window_size]

                R[ind, i, j] = compute_corner_response(I_x, I_y)
        # thresholding of R
        R[ind] = np.abs(R[ind])
        if False:
            x, y = np.where(
                R[ind] > np.percentile(R[ind], 75))
            x, y = from_R_to_im(x, y, window_size, stride)

            heatmap = R_to_heatmap(R[ind], window_size, stride, im_size)
            imshow(image_list[ind], points_of_interest=(x, y), heatmap=heatmap)
            continue
    np.savetxt("R"+str(time.time())+".txt", R.reshape((image_list.shape[0],
                                                       R_size**2)))

    for ind in range(image_list.shape[0]):
        x, y = np.where(
            R[ind] > np.percentile(R[ind], 75))
        x, y = from_R_to_im(x, y, window_size, stride)
        zipped = zip(x, y)
        for x, y in zipped:  # r_pin are the coordinates in R of POI
            patch_x = image_grad_x[x-patch_size:x+patch_size,
                                   y-patch_size:y+patch_size]
            patch_y = image_grad_y[x-patch_size:x+patch_size,
                                   y-patch_size:y+patch_size]
            pin_as_matrix = discretize_orientation(patch_x, patch_y)
            pin = pin_as_vect(pin_as_matrix)
            pins.append(pin)

    # KMEANS


def from_R_to_im(x, y, window_size, stride):
    x = x * stride + (window_size - 1)//2 + (FILTER_SIZE - 1)//2
    y = y * stride + (window_size - 1)//2 + (FILTER_SIZE - 1)//2

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
