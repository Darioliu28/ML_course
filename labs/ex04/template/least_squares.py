# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_loss(y, tx, w, method):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e=y-tx@w
    N=y.shape[0]
    if method=="MSE":
        L=(1/(2*N))*np.dot(e,e)
    elif method=="MAE":
        L=(1/N)*np.sum(np.abs(e))
    else:
        raise ValueError("method must be 'MSE' or 'MAE'")

    return L

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """

    G=tx.T@tx
    b=tx.T@y
    w=np.linalg.solve(G,b)
    mse=compute_loss(y,tx,w,method="MSE")

    return w, float(mse)