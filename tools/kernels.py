import numpy as np


def kernel(x, y, type="linear", **kwargs):
    '''
    Args :
           - x ndarray (., d): sample image.
           - y ndarray (., d): sample image.
           - type string: which type of kernel you want to derive.
           - **kwargs: arguments to be passed on to a specific kernel type.
    Returns :
             - float: the value of k(x,y)
    '''
    if type == "linear":
        return np.dot(x, y)
    else:
        raise ValueError("The {} kernel is not implemented".format(type))


def kernel_matrix(X, type="linear", **kwargs):
    '''Computes the kernel matrix for the input data X.
        Arguments:
            - X (ndarray): the input data.
            - type (str): the type of kernel you want to use.
            - **kwargs: arguments to be passed on to a specific kernel type.
        Returns:
            - ndarray: the kernel matrix.
    '''
    n_data = X.shape[0]
    K = np.zeros((n_data, n_data))
    for i in range(n_data):
        for j in range(i+1):
            K[i, j] = kernel(X[i, :], X[j, :], type=type, **kwargs)
            K[j, i] = K[i, j]
    return K
