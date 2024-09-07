#!/usr/bin/env python3
""" Simple Bidirectional of a Recurrent Neural Network

Requires:
    - numpy
"""

import numpy as np


class BidirectionalCell:
    """ Represents a Bidirectional cell of an RNN """
    def __init__(self, i, h, o):
        """ Initializes a Bidirectional RRN Cell
            - i: dimensionality of the data
            - h: dimensionality of the hidden state
            - o: dimensionality of the outputs
        """
        # Weights
        self.Whf = np.random.normal(size=(i + h, h))  # Hidden state forward
        self.Whb = np.random.normal(size=(i + h, h))  # Hidden state backward
        self.Wy = np.random.normal(size=(h * 2, o))   # Cell output

        # Biases
        self.bhf = np.zeros(shape=(1, h))             # Hidden state fowrward
        self.bhb = np.zeros(shape=(1, h))             # Hidden state backward
        self.by = np.zeros(shape=(1, o))              # Cell output

    def _softmax(self, x):
        """ Calculates softmax activation function for an given x """
        expx = np.exp(x)

        return expx / np.sum(expx, axis=1, keepdims=True)

    def _sigmoid(self, x):
        """ Calculates the sigmoid activation function for a given x """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """ calculates the hidden state in the forward direction
            for one time step
            - h_prev: ndarray, shape(m, h) previous hidden sate
            - x_t: ndarray, shape(m, i) data input
        """
        xprevxt = np.concatenate((h_prev, x_t), axis=1)

        return np.tanh(xprevxt @ self.Whf + self.bhf)
