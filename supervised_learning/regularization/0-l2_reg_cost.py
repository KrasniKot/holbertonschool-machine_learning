#!/usr/bin/env python3
""" This module contains l2_reg_cost(),
    that calculates the cost of a neural network with L2 regularization

    requires:
        - numpy.
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Calculates the cost of a NN with L2 regularization.
        - cost: cost of the network without L2 regularization,
        - lambtha: regularization parameter,
        - weights: dictionary of the weights and biases
            (numpy.ndarrays) of the neural network,
        - L: number of layers in the neural network,
        - m: number of data points used.
    """
    n = 0

    for i in range(1, L + 1):
        w = weights["W" + str(i)]  # Exctract the weights for current layer
        n += np.sum(np.square(w))  # Add up all the squared weights

    n *= lambtha / (2 * m)  # Scale n

    return cost + n
