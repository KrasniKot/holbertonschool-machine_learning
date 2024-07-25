#!/usr/bin/env python3
""" This module contains correlation(),
    that returns the correlation
    for the given covariance matrix
"""

import numpy as np


def correlation(C):
    """ Calculates a correlation matrix
        - C: numpy.ndarray of shape (d, d) that contains a covariance matrix
            - d: number of dimensions
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")
    d, d_2 = C.shape
    if d != d_2:
        raise ValueError("C must be a 2D square matrix")

    D = np.sqrt(np.diag(C))

    return 1 / np.outer(D, D) * C
