#!/usr/bin/env python3
""" This module contains normalization_constants(),
    which returns the mean and the standard deviation of a given matrix,

    requires: numpy
"""

import numpy as np


def normalization_constants(X):
    """ Calculates the standardization constants of a matrix;
        - X: np.ndarray of shape (m, nx).
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
