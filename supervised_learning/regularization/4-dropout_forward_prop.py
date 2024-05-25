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

    - X: Input data for the network
        * nx: number of input features
        * m: number of data points
    - weights: Weights and biases of the network
    - L: Number of layers in the network
    - keep_prob: Probability that a node will be kept

    """
    a = {}  # Stores the outputs of each layer
    a["A0"] = X  # Input for the network

    for layer in range(L):
        weight = weights["W{}".format(layer + 1)]
        bias = weights["b{}".format(layer + 1)]

        z = np.matmul(weight, a[f"A{layer}".format(layer)]) + bias  # Output for current layer

        dropout = np.random.binomial(1, keep_prob, size=z.shape)  # Generate a binary mask of 1s and 0s, where the amount of 1s depends on keep_prob

        if layer != (L - 1):  # If not output layer
            A = np.tanh(z)  # Apply tanh activation
            A *= dropout  # Apply dropout
            A /= keep_prob  # Scale output
            a["D{}".format(layer + 1)] = dropout

        else:
            A = np.exp(z)
            A /= np.sum(A, axis=0, keepdims=True)  # Apply softmax activation
    
        a["A{}".format(layer + 1)] = A

    return a
