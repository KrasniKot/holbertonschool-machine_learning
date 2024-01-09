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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for la in range(self.L):

            if type(layers[la]) is not int or layers[la] < 1:
                raise ValueError("layers must be a list of positive integers")
            
            if la == 0:
                self.weights["W" + str(la + 1)] = np.random.randn(layers[la], nx) * np.sqrt(2 / nx)

            if la > 0:
                self.weights["W" + str(la + 1)] = np.random.randn(layers[la], layers[la - 1]) * np.sqrt(2 / layers[la - 1])

            self.weights["b" + str(la + 1)] = np.zeros((layers[la], 1))

