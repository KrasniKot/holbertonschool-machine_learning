#!/usr/bin/env python3
"""
This module contains the class NeuralNetwork
which performs binary classification
"""

import numpy as np


class NeuralNetwork():
    """ Defines a Neural Network """

    def __init__(self, nx, nodes):
        """ Initializes a neural Network
                - nx: number of input features.
                - nodes: number of nodes in the hidden layer
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
