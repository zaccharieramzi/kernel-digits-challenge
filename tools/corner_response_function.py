import numpy as np


def gaussian_filter_2d(shape=(3, 3), sigma=0.5):
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


GAUSSIAN_FILTER = gaussian_filter_2d()


def compute_gaussian_grad(image_mat):
    '''
    args :
        - image_mat nd array (image_size, image_size)
    returns :
        image_grad_x DoG x
        image_grad_y DoG y
    '''
    image_grad_x = np.zeros_like(image_mat)
    image_grad_y = np.zeros_like(image_mat)

    def grad(im):
        fx = np.zeros_like(im)
        fy = np.zeros_like(im)
        fx[:-1, :] = im[1:, :] - im[:-1, :]
        # fx[-1, :] = 0 redundant

        fy[:, :-1] = im[:, 1:] - im[:, :-1]
        # fy[:, -1] = 0 redundant
        return fx, fy

    filterx, filtery = grad(GAUSSIAN_FILTER)

    for i in range(image_mat.shape[0]):
        for j in range(image_mat.shape[1]):
            rmin = max(0, i - filterx.shape[0] // 2)
            rmax = min(image_mat.shape[0], i + filterx.shape[0] // 2)
            rmin_f = rmin - i + filterx.shape[0] // 2
            rmax_f = rmax - i + filterx.shape[0] // 2
            if rmax_f == 0:
                rmax_f = filterx.shape[0]

            smin = max(0, j - filterx.shape[1] // 2)
            smax = min(image_mat.shape[1], j + filterx.shape[1] // 2)
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







    return image_grad_x, image_grad_y
