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
            return np.linalg.solve((K + lamb * n * np.eye(n)), Y)
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
                Y = 2 * Y - 1
                alpha = np.zeros(n)
                W = np.eye(n)
                z = np.zeros(n)
                for iter in range(n_iter):
                    alpha = solveWKRR(K, W, z, lamb, n)
                    m = K.dot(alpha)
                    W = np.diag(sigma(m) * sigma(-m))
                    z = m + Y / sigma(-Y * m)
                return alpha
    elif prob_type == "svm":
        try:
            lamb = kwargs["lamb"]
        except KeyError:
            raise KeyError("You need a lamb argument when performing an svm")
        else:
            n_iter = kwargs.get("n_iter", 10000)
            return svm(K, Y, lamb, n_iter)
    elif prob_type == "fast_svm":
        try:
            lamb = kwargs["lamb"]
        except KeyError:
            raise KeyError(
                "You need a lamb argument when performing a fast_svm")
        else:
            n_iter = kwargs.get("n_iter", 10000)
            return fast_svm(K, Y, lamb, n_iter)
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
    Y_svm = 2 * Y - 1
    n = K.shape[0]
    alpha = np.zeros(n)
    for i in range(n_iter):
        j = randint(0, n - 1)
        beta = Y_svm[j] + K[j, j] * alpha[j] - np.dot(alpha, K[:, j])
        beta /= K[j, j]
        if Y_svm[j] * beta < 0:
            alpha[j] = 0
        elif Y_svm[j] * beta < 1 / (2 * lamb * n):
            alpha[j] = beta
        else:
            alpha[j] = Y_svm[j] / (2 * lamb * n)
    return alpha


def fast_svm(K, Y, lamb, n_iter=10000):
    '''Solving the SVM quadratic problem with a double coordinate descent.
        Args:
            - K (ndarray): the kernel matrix of the observations.
            - Y (ndarray): the labels of the observations.
            - lamb (float): the regularization parameter.
            - n_iter (int): the number of iterations for the coordinate
            descent.
    '''
    Y_svm = 2 * Y - 1
    n = K.shape[0]
    alpha = np.zeros(n)
    for t in range(n_iter):
        j = randint(0, n - 1)
        i = randint(0, n - 1)
        while i == j:
            i = randint(0, n - 1)

        a_i = Y_svm[i] - np.dot(alpha, K[:, i])
        a_i += alpha[i] * K[i, i] + alpha[j] * K[j, i]
        a_j = Y_svm[j] - np.dot(alpha, K[:, j])
        a_j += alpha[i] * K[i, j] + alpha[j] * K[j, j]
        a = np.array([a_i, a_j])
        K_sub = np.array([
            [K[i, i], K[i, j]],
            [K[i, j], K[j, j]]
        ])
        cand_alph_i, cand_alph_j = np.linalg.solve(K_sub, a)
        # Update for i coordinate: we project the candidate on the
        # restrictions.
        if Y_svm[i] * cand_alph_i < 0:
            alpha[i] = 0
        elif Y_svm[i] * cand_alph_i < 1 / (2 * lamb * n):
            alpha[i] = cand_alph_i
        else:
            alpha[i] = Y_svm[i] / (2 * lamb * n)
        # Update for j coordinate.
        if Y_svm[j] * cand_alph_j < 0:
            alpha[j] = 0
        elif Y_svm[j] * cand_alph_j < 1 / (2 * lamb * n):
            alpha[j] = cand_alph_j
        else:
            alpha[j] = Y_svm[j] / (2 * lamb * n)
    return alpha


def svm_intercept(K, Y, lamb, n_iter=10000):
    '''Solving the SVM with intercept quadratic problem with a coordinate
    descent.
        Args:
            - K (ndarray): the kernel matrix of the observations.
            - Y (ndarray): the labels of the observations.
            - lamb (float): the regularization parameter.
            - n_iter (int): the number of iterations for the coordinate
            descent.
    '''
    Y_svm = 2 * Y - 1
    n = K.shape[0]
    alpha = np.zeros(n)
    for t in range(n_iter):
        j = randint(0, n - 1)
        i = randint(0, n - 1)
        while i == j:
            i = randint(0, n - 1)
        if Y_svm[i] * Y_svm[j] == -1:
            if Y_svm[i] == -1:
                # We switch the variables in order to have less cases
                temp = i
                i = j
                j = temp
        alpha_sum_w_ij = np.sum(alpha) - alpha[i] - alpha[j]
        alpha_bound = 1 / (2 * lamb * n)
        beta = alpha_sum_w_ij
        beta *= K[i, j] - K[j, j]
        beta += Y_svm[i] - Y_svm[j]
        beta /= K[i, i] + K[j, j] - 2 * K[i, j]
        if Y[i] == 1 and Y[j] == 1:
            if beta < max(0, -(alpha_sum_w_ij + alpha_bound)):
                alpha[i] = max(0, -(alpha_sum_w_ij + alpha_bound))
            elif beta < min(alpha_bound, -alpha_sum_w_ij):
                alpha[i] = beta
            else:
                alpha[i] = min(alpha_bound, -alpha_sum_w_ij)
        elif Y[i] == -1 and Y[j] == -1:
            if beta < max(-alpha_bound, -alpha_sum_w_ij):
                alpha[i] = max(-alpha_bound, -alpha_sum_w_ij)
            elif beta < min(0, alpha_bound - alpha_sum_w_ij):
                alpha[i] = beta
            else:
                alpha[i] = min(0, alpha_bound - alpha_sum_w_ij)
        else:
            if beta < max(0, -alpha_sum_w_ij):
                alpha[i] = max(0, -alpha_sum_w_ij)
            elif beta < min(alpha_bound, alpha_bound - alpha_sum_w_ij):
                alpha[i] = beta
            else:
                alpha[i] = min(alpha_bound, alpha_bound - alpha_sum_w_ij)
        alpha[j] = - alpha[i] - alpha_sum_w_ij
    return alpha


def sigma(X):
    '''
    sigmoid function
    Args :
           - X ndarray (., 1): a one dimension vector

    Returns :
             - the sigmoid of the vector ndarray (., 1)
    '''
    return 1 / (1 + np.exp(-X))


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
    I = lamb * n * np.eye(n)
    return W_sqrt.dot(np.linalg.solve((W_sqrt.dot(K).dot(W_sqrt) + I),
                                      W_sqrt.dot(z)))
