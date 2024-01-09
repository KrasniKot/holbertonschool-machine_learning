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

        for layer in range(1, self.L + 1):

            if type(layer) is not int:
                raise ValueError("layers must be a list of positive integers")

            self.weights['W' + str(layer)] = np.random.randn(
                    layers[layer - 1], nx) * np.sqrt(2 / nx)
            self.weights['b' + str(layer)] = np.zeros((layers[layer - 1], 1))
