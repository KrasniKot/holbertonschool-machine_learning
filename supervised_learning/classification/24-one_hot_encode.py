#!/usr/bin/env python3
""" This module contains one_hot_encoder() """

import numpy as np


def one_hot_encode(Y, classes):
    """ Converts a numeric label vector into a one-hot matrix.
        - Y: numpy.ndarray with shape (m,) containing numeric class labels.
        - classes: is the maximum number of classes found in Y.
    """
    try:
        return np.eye(classes)[Y].T
    except Exception:
        return None
