import numpy as np

from .corner_response_function import compute_gaussian_grad, compute_corner_response
from .visualization import reshape_as_images, imshow
from .data_loading import load_images


def feature_learning():
    X = load_images(type="test")[:100]
    image_list = reshape_as_images(X)
    image_list = image_list.mean(axis=3)
    window_size = 3
    stride = 3
    patch_size = 5
    pins = list()
    for ind in range(image_list.shape[0]):
        image_mat = image_list[ind]
        im_size = image_mat.shape[0]  # normally 32
        image_grad_x, image_grad_y = compute_gaussian_grad(image_mat)

        R_size = (im_size - window_size)/stride + 1
        R = np.zeros((R_size, R_size))
        for i in range(R_size):
            for j in range(R_size):
                I_x = image_grad_x[i*stride:i*stride+window_size]
                I_y = image_grad_y[j*stride:j*stride+window_size]
                R[i, j] = compute_corner_response(I_x, I_y)
        # thresholding of R
        R = np.abs(R)
        x, y = np.where(
            R > np.percentile(R, 85))
        x = (x+1) * stride
        y = (y+1) * stride
        imshow(X[ind], points_of_interest=(x, y))
        continue
        threshold = 1
        R = R > threshold
        # data viz / number of point of interest
        # ex : R > np.percentile(R, 90) -> top 10%


        for x_r, y_r in np.where(R):  # r_pin are the coordinates in R of point of interest
            im_pin = from_R_to_im(r_pin, window_size, stride, im_size)
            patch_x = image_grad_x[1]
            patch_y = image_grad_y[1]
            angles = compute_orientation(patch_x, patch_y)
            pin = discretize_orientation(angles)
            pins.append(pin)

    # KMEANS
