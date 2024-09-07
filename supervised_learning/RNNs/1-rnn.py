#!/usr/bin/env python3
""" Forward propagation for a RNN

Requires:
    - numpy
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """ Performs forward propagation for a Recurrent Neural Network
        - rnn_cell: A RNNCell instance
        - X: data to be used, numpy.ndarray of shape (t, m, i)
             - t is the maximum number of time steps
             - m is the batch size
             - i is the dimensionality of the data
        - h_o: initial hidden state, numpy.ndarray of shape (m, h)
    """

    t, m, i = X.shape
    _, h = h_0.shape

    # initialization of h_prev and y
    h_prev = h_0
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))

    # Compute output and hidden state for each time
    for i in range(t):
        h_next, y = rnn_cell.forward(h_prev, X[i])
        h_prev = h_next
        H[i + 1] = h_next
        Y[i] = y

    return H, Y
