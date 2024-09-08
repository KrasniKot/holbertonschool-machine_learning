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
    h_forward = h_0
    for step in range(t):
        h_forward = bi_cell.forward(h_forward, X[step])     # Forward cell pass
        # The hidden state is stored for the time t, for all the forward
        #    hidden states
        H[step, :, :h] = h_forward
    
    # Backward pass
    h_backward = h_t
    for step in reversed(range(t)):
        h_backward = bi_cell.backward(h_backward, X[step])  # Backward cell pass
        # The hidden state is stored for the time t, for all the backard
        #    hidden states
        H[step, :, h:] = h_backward
    
    # Compute outputs
    Y = bi_cell.output(H)
    
    return H, Y
