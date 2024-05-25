#!/usr/bin/env python3
""" This module contains dropout_forward_prop(),
    that conducts forward propagation using Dropout.

    requires:
        numpy.
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

        X [numpy.ndarray of shape(nx, m)]:
            contains the input data for the network
            nx: number of input features
            m: number of data points
        weights [dict]:
            contains weights and biases of the network
        L [int]:
            number of layers in the network
        keep_prob [float]:
            probability that a node will be kept

    """
    outputs = {}
    outputs["A0"] = X
    for index in range(L):
        weight = weights["W{}".format(index + 1)]
        bias = weights["b{}".format(index + 1)]
        z = np.matmul(weight, outputs["A{}".format(index)]) + bias
        dropout = np.random.binomial(1, keep_prob, size=z.shape)
        if index != (L - 1):
            A = np.tanh(z)
            A *= dropout
            A /= keep_prob
            outputs["D{}".format(index + 1)] = dropout
        else:
            A = np.exp(z)
            A /= np.sum(A, axis=0, keepdims=True)
        outputs["A{}".format(index + 1)] = A
    return outputs
