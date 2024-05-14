#!/usr/bin/env python3
""" This module contains shuffle_data(),
    which returns two shuffled matrices.

    requires: numpy.
"""

import numpy as np

per = np.random.permutation


def shuffle_data(X, Y):
    """ Shuffles two matrices.
        - X: np.ndarray of shape (m, nx) to shuffle.
        - Y:  np.ndarray of shape (m, ny) to shuffle.
    """
    s = per(len(X))
    return X[s], Y[s]
