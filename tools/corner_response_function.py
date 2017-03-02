import numpy as np


def compute_corner_response(I_x, I_y):
    '''
    args :
        - I_x ndarray gradient with regard to x
        - I_y ndarray gradient with regard to y
    return
        - R float corner Response for the window
    '''
    M = np.empty((2, 2))
    M[0, 0] = np.square(I_x).sum()
    M[0, 1] = np.sum(I_x * I_y)
    M[1, 0] = M[0, 1]
    M[1, 1] = np.square(I_y).sum()

    R = M[0, 0] * M[1, 1] - M[1, 0] * M[0, 1] - 0.05 * (M[0, 0] + M[1, 1])**2

    return R


def gaussian_filter_2d(shape=(3, 3), sigma=0.25):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def difference_of_Gaussian_filters(shape=(5, 5), sigma=0.7):
    # we compute the filter only once
    filtery, filterx = np.gradient(
        gaussian_filter_2d(shape=shape, sigma=sigma))
    return filterx, filtery


def convolve(image_mat, filterx, filtery):
    '''
    compute the difference of gradients along the x and y axis (two outputs)
    args :
        - image_mat nd array (image_size, image_size)
        - filterx nd array
        - filtery nd array
    returns :
        image_grad_x DoG x
        image_grad_y DoG y
    '''
    image_grad_x = np.zeros_like(image_mat)
    image_grad_y = np.zeros_like(image_mat)

    # convolution
    for i in range(image_mat.shape[0]):
        for j in range(image_mat.shape[1]):
            # x-axis indices for image
            rmin = max(0, i - filterx.shape[0] // 2)
            rmax = min(image_mat.shape[0], i + 1 + filterx.shape[0] // 2)
            # x-axis indices for filter
            rmin_f = rmin - i + filterx.shape[0] // 2
            rmax_f = rmax - i + filterx.shape[0] // 2
            if rmax_f == 0:
                rmax_f = filterx.shape[0]

            # y-axis indices for image
            smin = max(0, j - filterx.shape[1] // 2)
            smax = min(image_mat.shape[1], j + 1 + filterx.shape[1] // 2)
            # y-axis indices for filter
            smin_f = smin - j + filterx.shape[1] // 2
            smax_f = smax - j + filterx.shape[1] // 2
            if smax_f == 0:
                smax_f = filterx.shape[1]

            image_grad_x[i, j] = np.sum(
                image_mat[rmin:rmax, smin:smax] * filterx[rmin_f:rmax_f,
                                                          smin_f:smax_f])
            image_grad_y[i, j] = np.sum(
                image_mat[rmin:rmax, smin:smax] * filtery[rmin_f:rmax_f,
                                                          smin_f:smax_f])
    return image_grad_x, image_grad_y
