import numpy as np
from scipy.spatial.distance import cdist

from tools.kernels import kernel


def pred(X_train, X_pred, alpha, kernel_type="linear", **kwargs):
    '''Returns the image of each row of X_pred by the function defined by
    X_train, alpha and the kernel type (f = sum(alpha_i*K(x_i, .)) ).
        Args:
            - X_train (ndarray): the data on which you solved your optimization
            problem to obtain the alpha.
            - X_pred (ndarray): the data whose image you want.
            - alpha (ndarray): the solutions of the optimization problem.
            - kernel_type (str): the type of kernel you used in your
            optimization proble.
            - kwargs: additionnal arguments to be passed to the kernel.
        Output:
            - ndarray: the images of each row of X_pred for each alpha.
    '''
    n_train = X_train.shape[0]
    n_pred = X_pred.shape[0]
    n_alpha = alpha.shape[0]
    if kernel_type == "linear":
        K = kwargs.get("inner", X_pred.dot(X_train.T))
    elif kernel_type == "hellinger":
        X_pred = np.sqrt(X_pred)
        X_train = np.sqrt(X_train)
        K = X_pred.dot(X_train.T)
    elif kernel_type == "polynomial":
        try:
            deg = kwargs["degree"]
        except KeyError:
            raise KeyError(
                "You need a degree argument to compute a polynomial kernel")
        else:
            c = kwargs.get("constant", 0)
            inner = kwargs.get("inner", X_pred.dot(X_train.T))
            K = (inner + c)**deg
    elif kernel_type == "rbf":
        try:
            sigma = kwargs["sigma"]
        except KeyError:
            raise KeyError("You need a sigma argument to compute a Radial"
                           "Basis Function")
        else:
            pairwise_dists = kwargs.get(
                "dist", cdist(X_pred, X_train, 'euclidean'))
            K = np.exp(-pairwise_dists ** 2 / (2 * sigma ** 2))
    else:
        K = np.zeros((n_pred, n_train))
        for i in range(n_pred):
            for j in range(n_train):
                K[i, j] = kernel(
                    X_pred[i], X_train[j], kernel_type=kernel_type, **kwargs)
    Y = np.zeros((n_pred, n_alpha))
    for i in range(n_alpha):
        Y[:, i] = K.dot(alpha[i, :])
    return Y
