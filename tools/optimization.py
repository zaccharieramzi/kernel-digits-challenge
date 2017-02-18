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
    else:
        raise ValueError("{} is not implemented.".format(prob_type))
