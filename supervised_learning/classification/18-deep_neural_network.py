#!/usr/bin/env python3
"""
This module contains the class DeepNeuralNetwork
which performs binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """ Defines a Deep Neural Network """

    def __init__(self, nx, layers):
        """ Initializes a Deep Neural Nerwork
                - nx: number of input features.
                - layers: number of nodes in each layer of the network.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or not layers:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for la in range(self.L):

            if type(layers[la]) is not int or layers[la] < 1:
                raise ValueError("layers must be a list of positive integers")

            inodes = nx if la == 0 else layers[la - 1]

            self.__weights["W" + str(la + 1)] = np.random.randn(
                    layers[la], inodes) * np.sqrt(2 / inodes)

            self.__weights["b" + str(la + 1)] = np.zeros((layers[la], 1))

    @property
    def L(self):
        """ Returns L (array of number of nodes per layer) """
        return self.__L

    @property
    def cache(self):
        """ Returns cache """
        return self.__cache

    @property
    def weights(self):
        """ Returns weights """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation of the Neural Network
            - X: numpy.ndarray with shape (nx, m) containing the input data.
        """
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            ws = self.__weights["W" + str(i)]
            bs = self.__weights["b" + str(i)]
            A = self.__cache["A" + str(i - 1)]
            Z = np.dot(ws, A) + bs
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A" + str(i)] = A

        return A, self.__cache
