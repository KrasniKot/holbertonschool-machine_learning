#!/usr/bin/env python3
"""
This module contains the class NeuralNetwork
which performs binary classification
"""

import numpy as np


class NeuralNetwork:
    """ Defines a Neural Network """

    def __init__(self, nx, nodes):
        """ Initializes a neural Network
                - nx: number of input features.
                - nodes: number of nodes in the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ Returns W1 """
        return self.__W1

    @property
    def W2(self):
        """ Returns W2 """
        return self.__W2

    @property
    def b1(self):
        """ Returns b1 """
        return self.__b1

    @property
    def b2(self):
        """ Returns b2 """
        return self.__b2

    @property
    def A1(self):
        """ Returns A1 """
        return self.__A1

    @property
    def A2(self):
        """ Returns A2 """
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propagation of the Neural Network
                     - X: numpy.ndarray with shape (nx, m)
                         containing the input features
        """
        if type(X) is not np.ndarray:
            raise TypeError("X must be a numpy.ndarray")
        if X.shape[0] != self.W1.shape[1]:
            raise ValueError("Invalid number of features in X")

        def sigmoid(z): return 1 / (1 + np.exp(-z))

        self.__A1 = sigmoid(self.__W1 @ X + self.__b1)
        self.__A2 = sigmoid(self.__W2 @ self.__A1 + self.__b2)

        return self.__A1, self.__A2
