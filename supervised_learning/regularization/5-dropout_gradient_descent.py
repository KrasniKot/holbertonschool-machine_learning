#!/usr/bin/env python3
"""
Defines function that updates the weights with Dropout regularization
using gradient descent
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ Updates the weights of a NN with Dropout regularization using GD

     - Y: one-hot numpy.ndarray of shape (classes, m), contains
         the correct labels for the data
         * classes is the number of classes
         * m is the number of data points
     - weights: dictionary of the weights and biases of the NN
     - cache: dictionary of the outputs and dropout masks of
        each layer of the NN
     - alpha: Learning rate
     - keep_prob: Pobability that a node will be kept
     - L: Number of layers of the network
    """
    m = Y.shape[1]
    Al = cache['A' + str(L)]
    dAl = Al - Y

    for layer in reversed(range(1, L + 1)):
        w_key = 'W' + str(layer)
        b_key = 'b' + str(layer)
        Al_key = 'A' + str(layer)
        Al1_key = 'A' + str(layer - 1)
        D_key = 'D' + str(layer)

        Al = cache[Al_key]
        gld = 1 - np.power(Al, 2)
        if layer == L:
            dZl = dAl
        else:
            dZl = dAl * gld
            dZl *= cache[D_key] / keep_prob

        Wl = weights[w_key]
        Al1 = cache[Al1_key]
        dWl = (1 / m) * np.matmul(dZl, Al1.T)
        dbl = (1 / m) * np.sum(dZl, axis=1, keepdims=True)
        dAl = np.matmul(Wl.T, dZl)
        weights[w_key] = weights[w_key] - alpha * dWl
        weights[b_key] = weights[b_key] - alpha * dbl
