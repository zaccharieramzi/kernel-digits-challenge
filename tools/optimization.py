import numpy as np


def find_f(K, Y, prob_type="linear regression", **kwargs, n_iter=200):
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
                           linear regression")
        else:
            n = len(Y)
            Y = 2*Y - 1
            alpha = np.zeros(n)
            W = np.eye(n)
            P = 1/2*np.eye(n)
            z = np.zeros(n)
            for iter in range(n_iter):
                alpha = solveWKRR(K, W, z)
                m = K.dot(alpha)
                P = -sigma(-Y*m)
                W = sigma(m)*sigma(-m)
                z = m + Y/sigma(-Y*m)
            return alpha
    else:
        raise ValueError("{} is not implemented.".format(prob_type))


def sigma(X):
    '''
    sigmoid function
    Args :
           - X ndarray (., 1): a one dimension vector

    Returns :
             - the sigmoid of the vector
    '''
    return 1/(1+np.exp(-X))


def solveWKRR(K, W, z):
    '''
    solve a weighted linear regression problem for the logistic regression
    purpose
    Args :
           - K ndarray (., .): the kernel matrix
           - W ndarray (., .): the weights. Here the hessian of the sigmoid

    Returns :
             - the solution alpha at each step
    '''
    W_sqrt = np.sqrt(W)
    I = lamb*n*np.eye(n)
    return W_sqrt.dot(np.linalg.solve((W_sqrt.dot(K).dot(W_sqrt) + I,
                                      W_sqrt.dot(Y))))
