#!/usr/bin/env python3
""" Simple RNN cell

Requires:
    - numpy
"""

import numpy as np


class RNNCell:
    """ Represents a simple Recurrent Neural Network Cell """

    def __init__(self, i, h, o):
        """ Initializes a Simple RNN Cell
            - i is the dimensionality of the data
            - h is the dimensionality of the hidden state
            - o is the dimensionality of the outputs
        """
        self.Wh = np.random.normal(size=(i + h, h))  # Hidden state weights
        self.Wy = np.random.normal(size=(h, o))  # Ouput weights
        self.bh = np.zeros(shape=(1, h))  # Hidden state bias
        self.by = np.zeros(shape=(1, o))  # Hidden state output

    def forward(self, h_prev, x_t):
        """ Performs the forward propagation step for a RNN
            - h_prev: numpy.ndarray of shape (m, h),
                      contains the previous hidden state
            - x_t: numpy.ndarray of shape (m, i),
                   contains the data input for the cell
        """
        # Concatenated previous hidden state and input at time t
        #     so that they can be processed using a single weight matrix
        hxt = np.concatenate((h_prev, x_t), axis=1)

        # Next hidden state with tanh activation
        h_next = np.tanh((hxt @ self.Wh + self.bh))

        #              cell output
        return h_next, self.__softmax(h_next @ self.Wy + self.by)

    def __softmax(self, x):
        """ Calculates softmax activation function for an input x """
        expx = np.exp(x)

        return expx / np.sum(expx, axis=1, keepdims=True)
