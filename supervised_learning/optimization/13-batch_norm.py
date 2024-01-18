#!/usr/bin/env python3
""" This module contains batch_norm(),
    which normalizes an unactivated output of a nn using batch normalization.

    requires:
        - numpy.
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ Normalizes an unactivated output of a nn using batch normalization.
        - Z: np.ndarray to be normalized,
        - gamma: np.ndarray of shape (1, n),
            scales used for batch normalization,
        - beta: numpy.ndarray of shape (1, n),
            offsets used for batch normalization,
        - epsilon: small number to avoid division by zero.
    """
    mean = np.sum(Z, axis=0) / Z.shape[0]
    var = np.sum(np.power(Z - mean, 2), axis=0) / Z.shape[0]
    normz = (Z - mean) / np.sqrt(var + epsilon)

    return gamma * normz + beta
