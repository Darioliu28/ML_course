# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w,method):
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
