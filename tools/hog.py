import numpy as np

from .corner_response_function import convolve,\
    difference_of_Gaussian_filters
from .discretization import discretize_orientation, pin_as_vect


def hog(
        image,
        filter_sigma=0.1, filter_shape=5, hog_cell_size=8, disc_grid=16,
        normalize=False, block_size=2, color_grad=False):
    '''Computes histogram of gradients for a given image.
    Args:
        - image (ndarray): image (im_size x im_size x 3) whose HoG you want.
        - filter_sigma (float): the gaussian filter sigma.
        - filter_shape (odd int): the size of the gaussian gradient you want to
        compute.
        - hog_cell_size (int divisor of im_size): the size of the cells in
        which you want to have a histogram.
        - disc_grid (even int): pi/disc_grid will be the angle for
        discretization of the gradients' orientations.
        - normalize (bool): whether you want to normalize the histograms per
        block.
        - block_size (int): the square root of the number of cells in each
        normalization block.
        - color_grad (bool): whether you want to take into acocunt all the
        colors when computing the gradients. The gradient retained will be the
        one with biggest norm.
    Returns:
        - ndarray: ((im_size/hog_cell_size)**2 x disc_grid/2), the histograms
        of gradients flattened.
    '''
    im_size = image.shape[0]  # 32 or 63
    # gaussian filters
    filterx, filtery = difference_of_Gaussian_filters(
        shape=(filter_shape, filter_shape),
        sigma=filter_sigma)
    if color_grad:
        n_colors = image.shape[2]
        # image_grad_x represents the gaussian gradient of the image in x for
        # all three colors. However, for each the color, the gradient is
        # represented as a line to compute the best for each color.
        image_grad_x = np.zeros((n_colors, im_size**2))
        image_grad_y = np.zeros((n_colors, im_size**2))

        for c in range(n_colors):
            # we compute the gradient for each color
            grad_x, grad_y = convolve(image[:, :, c], filterx, filtery)
            image_grad_x[c, :] = grad_x.reshape((im_size**2,))
            image_grad_y[c, :] = grad_y.reshape((im_size**2,))

        image_grad_n = np.sqrt(image_grad_x**2 + image_grad_y**2)
        # we select only the biggest gradient thanks to some numpy magic.
        best_colors = np.argmax(image_grad_n, axis=0)
        image_grad_x = image_grad_x[
            best_colors, np.arange(im_size**2)].reshape((im_size, im_size))
        image_grad_y = image_grad_y[
            best_colors, np.arange(im_size**2)].reshape((im_size, im_size))
    else:
        X = image.mean(axis=2)
        image_grad_x, image_grad_y = convolve(X, filterx, filtery)
    ori, w = discretize_orientation(
        image_grad_x,
        image_grad_y,
        signed=False,
        disc_grid=16)
    n_cells = im_size // hog_cell_size
    # we compte in each cell the histogram of gradient
    cells_vector = np.zeros((n_cells, n_cells, disc_grid // 2))
    for i in range(n_cells):
        for j in range(n_cells):
            cells_vector[i, j, :] = pin_as_vect(
                discrete_or=ori[
                    i * hog_cell_size:(i + 1) * hog_cell_size,
                    j * hog_cell_size:(j + 1) * hog_cell_size
                ],
                weights=w[
                    i * hog_cell_size:(i + 1) * hog_cell_size,
                    j * hog_cell_size:(j + 1) * hog_cell_size
                ],
                disc_grid=disc_grid)
    if normalize:
        for i in range(n_cells // block_size):
            for j in range(n_cells // block_size):
                cells_vector[
                    i * block_size:(i + 1) * block_size,
                    j * block_size:(j + 1) * block_size
                ] /= np.linalg.norm(cells_vector[
                    i * block_size:(i + 1) * block_size,
                    j * block_size:(j + 1) * block_size
                ])
    return cells_vector.reshape((n_cells**2, disc_grid // 2))
