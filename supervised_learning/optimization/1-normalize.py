#!/usr/bin/env python3
""" This module contains normalize(),
    which standardizes a matrix.

    requires: numpy
"""

import numpy as np


def normalize(X, m, s):
    """ Standardizes a matrix.
        - X: np.ndarray of shape (d, nx) to normalize,
        - m: np.ndarray of shape (nx, ) containing the mean of all X features,
        - s: numpy.ndarray of shape (nx,) containing o of all X features,
    """
    return (X - m) / s
