#!/usr/bin/env python3
"""
This module contains the class Neuron
which performs binary classification
"""

import numpy as np


class Neuron():
    """ Defines a Neuron """

    def __init__(self, nx):
        """ Initializes a neuron;
                nx: number of input features
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Returns __W"""
        return self.__W

    @property
    def b(self):
        """ Returns __b """
        return self.__b

    @property
    def A(self):
        """ Returns __A """
        return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron
            X: Matrix with shape (nx, m);
                nx: number of input features
                m: number of examples
        """
        Z = self.__W @ X + self.b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A


    def cost(self, Y, A):
        """ Calculates the binary cross-entropy cost.
            - Y: Matrix with shape (1, m) containing true labels.
            - A: Matrix with shape (1, m) containing the activated output for each example
                - m: number of examples
        """
        m = Y.shape[1]
        c = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return c
