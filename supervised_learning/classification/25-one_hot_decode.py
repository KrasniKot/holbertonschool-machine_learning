#!/usr/bin/env python3
""" This module contians one_hot_decode() which decodes a one_hot matrix """
import numpy as np


def one_hot_decode(one_hot):
    """ Decodes a one_hot matrix into a vector of labels
        - one_hot: numpy.ndarray (classes, m) containing the matrix to decode
    """
    if type(one_hot) is np.ndarray and len(one_hot.shape) == 2:
        return one_hot.T.argmax(axis=1)
