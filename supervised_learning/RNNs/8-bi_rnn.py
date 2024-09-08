#!/usr/bin/env python3
""" Forward propagations for a bidirectional RNN cell

Requires:
    - numpy
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ Performs forward propagation for a bidirectional RNN.
        - bi_cell: Instance of BidirectionalCell used for forward propagation.
        - X: Data to be used for forward propagation, shape (t, m, i).
        - h_0: Initial hidden state in the forward direction, shape (m, h).
        - h_t: Initial hidden state in the backward direction, shape (m, h).
    """
    t, m, i = X.shape
    _, h = h_0.shape             # Dimensionality of the hidden state
    o = bi_cell.Wy.shape[1]       # Dimensionality of the outputs

    H = np.zeros((t, m, 2 * h))  # To store concatenated hidden states
    Y = np.zeros((t, m, o))      # To store outputs

    # Forward pass
    hforw = h_0
    for step in range(t):
        hforw = bi_cell.forward(hforw, X[step])     # Forward cell pass
        # The hidden state is stored for the time t, for all the forward
        #    hidden states
        H[step, :, :h] = hforw

    # Backward pass
    hback = h_t
    for step in reversed(range(t)):
        hback = bi_cell.backward(hback, X[step])    # Backward cell pass
        # The hidden state is stored for the time t, for all the backard
        #    hidden states
        H[step, :, h:] = hback

    # Compute outputs
    Y = bi_cell.output(H)

    return H, Y
