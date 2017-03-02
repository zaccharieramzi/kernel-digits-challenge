from random import randint

import numpy as np


def find_f(K, Y, prob_type="linear regression", **kwargs):
    '''
    Args :
           - K ndarray (., .): the kernel matrix
           - Y ndarray (.,): the labels (0 or 1)
           - prob_type: which type of classification problem you want to solve
           - **kwargs: arguments to be passed to the optimization solver
    Returns :
             - alpha
    '''
    if prob_type == "linear regression":
        try:
            lamb = kwargs["lamb"]
        except KeyError:
            raise KeyError("You need a lamb argument when performing a \
                           linear regression")
        else:
            n = len(Y)
            return np.linalg.solve((K + lamb*n*np.eye(n)), Y)
    elif prob_type == "logistic regression":
        try:
            lamb = kwargs["lamb"]
        except KeyError:
            raise KeyError("You need a lamb argument when performing a \
                           logistic regression")
        else:
            try:
                n_iter = kwargs["n_iter"]
            except KeyError:
                raise KeyError("You need a n_iter argument when performing a \
                               logistic regression")
            else:
                n = len(Y)
                Y = 2*Y - 1
                alpha = np.zeros(n)
                W = np.eye(n)
                z = np.zeros(n)
                for iter in range(n_iter):
                    alpha = solveWKRR(K, W, z, lamb, n)
                    m = K.dot(alpha)
                    W = np.diag(sigma(m)*sigma(-m))
                    z = m + Y/sigma(-Y*m)
                return alpha
    elif prob_type == "svm":
        try:
            lamb = kwargs["lamb"]
        except KeyError:
            raise KeyError("You need a lamb argument when performing an svm")
        else:
            n_iter = kwargs.get("n_iter", 10000)
            return svm(K, Y, lamb, n_iter)
    else:
        raise ValueError("{} is not implemented.".format(prob_type))


def svm(K, Y, lamb, n_iter=10000):
    '''Solving the SVM quadratic problem with a coordinate descent.
        Args:
            - K (ndarray): the kernel matrix of the observations.
            - Y (ndarray): the labels of the observations.
            - lamb (float): the regularization parameter.
            - n_iter (int): the number of iterations for the coordinate
            descent.
    '''
    Y_svm = 2*Y - 1
    n = K.shape[0]
    alpha = np.zeros(n)
    alpha_new = np.ones(n)
    for i in range(n_iter):
        alpha_new = alpha
        j = randint(0, n - 1)
        beta = (Y_svm[j] + K[j, j]*alpha[j] - np.dot(alpha, K[:, j])) / K[j, j]
        if Y_svm[j] * beta < 0:
            alpha_new[j] = 0
        elif Y_svm[j] * beta < 1 / (2*lamb*n):
            alpha_new[j] = beta
        else:
            alpha_new[j] = Y_svm[j] / (2*lamb*n)
    return alpha_new


def sigma(X):
    '''
    sigmoid function
    Args :
           - X ndarray (., 1): a one dimension vector

    Returns :
             - the sigmoid of the vector ndarray (., 1)
    '''
    return 1/(1+np.exp(-X))


def solveWKRR(K, W, z, lamb, n):
    '''
    solve a weighted linear regression problem for the logistic regression
    purpose
    Args :
           - K ndarray (., .): the kernel matrix
           - W ndarray (., .): the weights. Here the hessian of the sigmoid
           - z ndarray (., .): play the role of Y
           - lamb : regularization parameter
    Returns :
             - the solution alpha at each step
    '''
    W_sqrt = np.sqrt(W)
    I = lamb*n*np.eye(n)
    return W_sqrt.dot(np.linalg.solve((W_sqrt.dot(K).dot(W_sqrt) + I),
                                      W_sqrt.dot(z)))
