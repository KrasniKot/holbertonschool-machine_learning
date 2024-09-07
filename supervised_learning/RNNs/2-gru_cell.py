#!/usr/bin/env python3
""" Simple GRU cell

Requires:
    - numpy
"""
import numpy as np


class GRUCell:
    """ Represent a Gated Recurrent Unit """

    def __init__(self, i, h, o):
        """ Initializes a GRU Cell
            - i: dimensionality of the date
            - h: dimensionality of the hidden state
            - o: dimensionality of the outputs
        """
        normal = np.random.normal
        zeros = np.zeros

        self.Wz = normal(size=(i + h, h))  # Weights for update gate
        self.Wr = normal(size=(i + h, h))  # Weights for reset gate
        self.Wh = normal(size=(i + h, h))  # Weights for candidate hidden state
        self.Wy = normal(size=(h, o))      # Weights for output

        self.bh = zeros(shape=(1, h))      # Biases for update gate
        self.bz = zeros(shape=(1, h))      # Biases for reset gate
        self.br = zeros(shape=(1, h))      # Biases for candidate hidden state
        self.by = zeros(shape=(1, o))      # Biases for ouput gate

    def _softmax(self, x):
        """ Calculates softmax activation function for an given x """
        expx = np.exp(x)

        return expx / np.sum(expx, axis=1, keepdims=True)

    def _sigmoid(self, x):
        """ Calculates the sigmoid activation function for a given x """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """ Performs forward propagation for one time step.
            - h_prev: (numpy.ndarray), previous hidden state, shape (m, h)
            - x_t: (numpy.ndarray), input data at time step t, shape (m, i)
        """
        sigmoid = self._sigmoid
        softmax = self._softmax

        # Update gate
        hprevxt = np.concatenate((h_prev, x_t), axis=1)
        zt = sigmoid(hprevxt @ self.Wz + self.bz)

        # Reset gate
        rt = sigmoid(hprevxt @ self.Wr + self.br)

        # Candidate hidden state
        rthprev = np.concatenate((rt * h_prev, x_t), axis=1)
        chs = np.tanh(rthprev @ self.Wh + self.bh)

        # Next hidden state
        hnext = zt * chs + (1 - zt) * h_prev

        # output with softmax activation
        y = softmax(hnext @ self.Wy + self.by)

        return hnext, y