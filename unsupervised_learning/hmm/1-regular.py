#!/usr/bin/env python3
""" Computes the steady state probabilities of a markov chain """

import numpy as np


def regular(P):
    """ Determines the steady state probabilities of a regular markov chain
        - P: square 2D numpy.ndarray of shape (n, n), the transition matrix
    """
    # check that P is the correct type and dimensions
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None

    # save value of n and check that P is square
    if P.shape[0] != P.shape[1]:
        return None

    if not (P > 0).all():
        return None

    A = P.T - np.eye(P.shape[0])

    # Add the normalization condition
    A = np.vstack([A, np.ones(P.shape[0])])
    b = np.zeros(P.shape[0])
    b = np.append(b, 1)

    # Solve the linear system
    return np.linalg.lstsq(A, b, rcond=None)[0].reshape(1, -1)
