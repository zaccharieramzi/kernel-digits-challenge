import numpy as np

from .corner_response_function import convolve,\
    difference_of_Gaussian_filters
from .discretization import discretize_orientation, pin_as_vect


def hog(
        image,
        filter_sigma=0.1, filter_shape=5, hog_cell_size=8, disc_grid=16,
        normalize=False, block_size=2):
    X = image.mean(axis=2)
    image_size = X.shape[0]  # 32 or 63
    # gaussian filters
    filterx, filtery = difference_of_Gaussian_filters(
        shape=(filter_shape, filter_shape),
        sigma=filter_sigma)
    image_grad_x, image_grad_y = convolve(X, filterx, filtery)

    ori, w = discretize_orientation(
        image_grad_x,
        image_grad_y,
        signed=False,
        disc_grid=16)
    n_cells = image_size // hog_cell_size
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
