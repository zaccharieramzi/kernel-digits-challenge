import numpy as np
from scipy.spatial.distance import pdist, squareform


def kernel(x, y, kernel_type="linear", **kwargs):
    '''
    Args :
           - x ndarray (., d): sample image.
           - y ndarray (., d): sample image.
           - kernel_type string: which type of kernel you want to derive.
           - **kwargs: arguments to be passed on to a specific kernel type.
    Returns :
             - float: the value of k(x,y)
    '''
    if kernel_type == "linear":
        return np.dot(x, y)
    elif kernel_type == "hellinger":
        return np.dot(np.sqrt(x), np.sqrt(y))
    elif kernel_type == "rbf":
        try:
            sigma = kwargs["sigma"]
        except KeyError:
            raise KeyError("You need a sigma argument to compute a Radial"
                           "Basis Function")
        else:
            return np.exp(-np.linalg.norm(x - y)**2 / sigma ** 2)
    elif kernel_type == "polynomial":
        try:
            deg = kwargs["degree"]
        except KeyError:
            raise KeyError(
                "You need a degree argument to compute a polynomial kernel")
        else:
            c = kwargs.get("constant", 0)
            return (np.dot(x, y) + c)**deg
    else:
        raise ValueError("The {} kernel is not implemented".format(
            kernel_type))


def kernel_matrix(X, kernel_type="linear", **kwargs):
    '''Computes the kernel matrix for the input data X.
        Arguments:
            - X (ndarray): the input data.
            - kernel_type (str): the type of kernel you want to use.
            - **kwargs: arguments to be passed on to a specific kernel type.
        Returns:
            - ndarray: the kernel matrix.
    '''
    if kernel_type == "linear":
        inner = kwargs.get("inner", X.dot(X.T))
        return inner
    elif kernel_type == "hellinger":
        X = np.sqrt(X)
        return X.dot(X.T)
    elif kernel_type == "rbf":
        try:
            sigma = kwargs["sigma"]
        except KeyError:
            raise KeyError("You need a sigma argument to compute a Radial"
                           "Basis Function")
        else:
            pairwise_dists = kwargs.get(
                "dist", squareform(pdist(X, 'euclidean')))
            return np.exp(-pairwise_dists ** 2 / (2 * sigma ** 2))
    elif kernel_type == "polynomial":
        try:
            deg = kwargs["degree"]
        except KeyError:
            raise KeyError(
                "You need a degree argument to compute a polynomial kernel")
        else:
            c = kwargs.get("constant", 0)
            inner = kwargs.get("inner", X.dot(X.T))
            return (inner + c)**deg
    else:
        n_data = X.shape[0]
        K = np.zeros((n_data, n_data))
        for i in range(n_data):
            for j in range(i + 1):
                K[i, j] = kernel(X[i, :], X[j, :],
                                 kernel_type=kernel_type, **kwargs)
                K[j, i] = K[i, j]
        return K
