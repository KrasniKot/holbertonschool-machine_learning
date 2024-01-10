#!/usr/bin/env python3
""" This module contians one_hot_decode() which decodes a one_hot matrix """
import numpy as np


def one_hot_decode(one_hot):
    """ Decodes a one_hot matrix into a vector of labels
        - one_hot: numpy.ndarray (classes, m) containing the matrix to decode
    """
    try:
        if type(one_hot) is np.ndarray:
            return np.argmax(one_hot, axis=0)
    except Exception:
        return None
