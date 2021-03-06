import numpy as np

from tools.kernels import kernel


def pred(X_train, X_pred, alpha, kernel_type="linear", **kwargs):
    '''Returns the image of each row of X_pred by the function defined by
    X_train, alpha and the kernel type (f = sum(alpha_i*K(x_i, .)) ).
        Args:
            - X_train (ndarray): the data on which you solved your optimization
            problem to obtain the alpha.
            - X_pred (ndarray): the data whose image you want.
            - alpha (ndarray): the solution ofthe optimization problem.
            - kernel_type (str): the type of kernel you used in your
            optimization proble.
            - kwargs: additionnal arguments to be passed to the kernel.
        Output:
            - ndarray: the images of each row of X_pred.
    '''
    n_train = X_train.shape[0]
    n_pred = X_pred.shape[0]
    K = np.zeros((n_pred, n_train))
    for i in range(n_pred):
        for j in range(n_train):
            K[i, j] = kernel(X_pred[i], X_train[j], kernel_type=kernel_type,
                             **kwargs)
    return K.dot(alpha)
