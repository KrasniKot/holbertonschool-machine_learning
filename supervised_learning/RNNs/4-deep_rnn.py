#!/usr/bin/env python3
""" Forward propagation for a deep Recurrent Neural Network

Requires:
    - numpy
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """ Performs forward propagation for a deep RNN

    -  rnn_cells: list of RNNCell instances of length 1
                  used for forward propagation
     - X: ndarray, shape(t, m, i), data to be used
          - t: maximum number of time steps
          - m: batch size
          - i: dimensionality of the data
     - h_0: ndarray, shape(l, m, h), initial hidden state
            - h: dimensionality of the hidden state
    """
    # Dimension extraction
    t, m, i = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].Wy.shape[1]

    # Initialiation for the lists containing the outputs and hidden states
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    # First of hidden states is the initial hidden state
    H[0] = h_0

    # Over each time step
    for step in range(t):
        # The input to the first layer at time step `step` is `X[step]`
        xt = X[step]

        # Over each layer in the deep RNN
        for layer in range(len(rnn_cells)):

            rnn_cell = rnn_cells[layer]  # Current RNN cell
            hprev = H[step, layer]       # Previous hidden state for this layer

            # Forward propagation for the current cell
            hnext, yt = rnn_cell.forward(hprev, xt)

            # The next hidden state is stored in H for the next time step
            H[step + 1, layer] = hnext

            # The output from the current layer is passed to the next layer
            xt = hnext

        # Store the output of the final layer at this time step
        Y[step] = yt

    return H, Y
