#!/usr/bin/env python3
""" Simple GRU cell

Requires:
    - numpy
"""

import numpy as np


class LSTMCell:
    """ Represents a Long-Short Term Memory Cell """

    def __init__(self, i, h, o):
        """ Initializes a GRU Cell
            - i: dimensionality of the date
            - h: dimensionality of the hidden state
            - o: dimensionality of the outputs
        """
        normal = np.random.normal
        zeros = np.zeros

        self.Wf = normal(size=(i + h, h))  # Forget gate weights
        self.Wu = normal(size=(i + h, h))  # Update gate weights
        self.Wc = normal(size=(i + h, h))  # Candidate cell state biases
        self.Wo = normal(size=(i + h, h))  # Output gate weights
        self.Wy = normal(size=(h, o))      # Cell's output weights

        self.bf = zeros(shape=(1, h))      # Forget gate biases
        self.bu = zeros(shape=(1, h))      # Update gate biases
        self.bc = zeros(shape=(1, h))      # Candidate cell state biases
        self.bo = zeros(shape=(1, h))      # Output gate biases
        self.by = zeros(shape=(1, o))      # Cell's ouput biases

    def _softmax(self, x):
        """ Calculates softmax activation function for an given x """
        expx = np.exp(x)

        return expx / np.sum(expx, axis=1, keepdims=True)

    def _sigmoid(self, x):
        """ Calculates the sigmoid activation function for a given x """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """ Performs forward propagation for one time step for the LSTM Cell
            - h_prev: ndarray, shape(m, h), previous hidden state
            - c_prev: ndarray, shape(m, h), previous cell state
            - x_t: ndarray, shape(m, i), data input
        """
        conc = np.concatenate

        # Concatenations in order to process previous states and inputs
        #    at a given time, simultaneously
        hprevxt = conc((h_prev, x_t), axis=1)

        ft = self._sigmoid(hprevxt @ self.Wf + self.bf)  # Forget gate
        ut = self._sigmoid(hprevxt @ self.Wu + self.bu)  # Update gate
        ccs = np.tanh(hprevxt @ self.Wc + self.bc)       # Candidate cell state

        ncs = ft * c_prev + ut * ccs                     # New cell state
        ot = self._sigmoid(hprevxt @ self.Wo + self.bo)  # Output Gate
        ht = ot * np.tanh(ncs)                           # Hidden state

        y = self._softmax(ht @ self.Wy + self.by)        # Cell's output

        return ht, ncs, y
