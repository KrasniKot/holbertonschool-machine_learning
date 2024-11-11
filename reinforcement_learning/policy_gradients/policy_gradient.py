#!/usr/bin/env python3
""" Have a function compute the policy of a matrix """

import numpy as np


def policy(matrix, weight):
    """ Computes the policy of a matrix
        - matrix ..... 2D array representing a row vector of features for a sin
        - weight ..... 2D matrix  with shape (state features, actions).
                       Each row corresponds to the weights applied to each feat
    """
    z = matrix @ weight  # Linear combination

    ez = np.exp(z - np.max(z, axis=1, keepdims=True))

    return ez / np.sum(ez, axis=1, keepdims=True)
