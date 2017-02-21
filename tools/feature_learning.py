import numpy as np

from .corner_response_function import compute_gaussian_grad, compute_corner_response
from .visualization import reshape_as_images, imshow
from .data_loading import load_images

FILTER_SIZE = 3


def feature_learning():
    X = load_images(type="test")[:100]
    image_list = reshape_as_images(X)
    image_list = image_list.mean(axis=3)
    window_size = 5
    stride = 3
    patch_size = 5
    pins = list()
    for ind in range(image_list.shape[0]):
        image_mat = image_list[ind]
        im_size = image_mat.shape[0]  # normally 32
        image_grad_x, image_grad_y = compute_gaussian_grad(image_mat)

        R_size = (image_grad_x.shape[0] - window_size)//stride + 1
        R = np.zeros((R_size, R_size))
        for i in range(R_size):
            for j in range(R_size):
                I_x = image_grad_x[i*stride:i*stride+window_size,
                                   j*stride:j*stride+window_size]
                I_y = image_grad_y[i*stride:i*stride+window_size,
                                   j*stride:j*stride+window_size]

                R[i, j] = compute_corner_response(I_x, I_y)
        # thresholding of R
        R = np.abs(R)
        x, y = np.where(
            R > np.percentile(R, 75))
        x, y = from_R_to_im(x, y, window_size, stride)

        heatmap = R_to_heatmap(R, window_size, stride, im_size)
        imshow(image_list[ind], points_of_interest=(y, x), heatmap=heatmap)
        continue
        threshold = 1
        R = R > threshold
        # data viz / number of point of interest
        # ex : R > np.percentile(R, 90) -> top 10%

        for x_r, y_r in np.where(R):  # r_pin are the coordinates in R of point of interest
            im_pin = from_R_to_im(r_pin, window_size, stride, im_size)
            patch_x = image_grad_x
            patch_y = image_grad_y
            angles = compute_orientation(patch_x, patch_y)
            pin = discretize_orientation(angles)
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
