#!/usr/bin/env python3
""" This module contains l2_reg_gradient_descent(),
    that updates the weights and biases of a neural network
    using gradient descent with L2 regularization.

    requires:
        - numpy.
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ Updates the weights and biases of a NN using GD with L2 regularization.
        - Y: one-hot np.ndarray of shape (classes, m)
            that contains the correct labels for the data,
        - weigts: dictionary of the weights and biases of the neural network,
        - cache: dictionary of the outputs of each layer of the neural network,
        - alpha: learning rate,
        - lambtha: L2 regularization parameter,
        - L: number of layers of the network;
            - m: Number of data points.
    """
    m = len(Y[1])

    dz = cache['A'+str(L)] - Y

    for i in range(L, 0, -1):

        L2 = (lambtha / m) * weights['W'+str(i)]
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dw = (1 / m) * (dz @ cache['A'+str(i-1)].T) + L2
        dz = (weights['W'+str(i)].T @ dz) * ((1 - cache['A'+str(i-1)] ** 2))

        weights['W'+str(i)] -= (alpha * dw)
        weights['b'+str(i)] -= (alpha * db)
