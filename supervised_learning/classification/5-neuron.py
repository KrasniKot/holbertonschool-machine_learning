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
            - A: Matrix with shape (1, m) containing the activated outputs
                - m: number of examples
        """
        m = Y.shape[1]
        c = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1.0000001-A))
        return c

    def evaluate(self, X, Y):
        """ Evaluates the neuron prediction and loss
                - X: numpy.ndarray with shape (nx, m) containing the inputs.
                - Y: numpy.ndarray with shape (1, m) containing true labels.
        """
        m = X.shape[1]

        A = self.forward_prop(X)
        L = np.where(A >= 0.5, 1, 0)
        c = self.cost(Y, A)

        return L, c

    def gradient_descent(self, X, Y, A, alpha=0.5):
        """ Perfomrs one step of gradient descent to update W and b
            - X: numpy.ndarray with shape (nx, m) containing input data.
                nx: number of input features
            - Y: numpy.ndarray with shape (1, m) containing true labels.
            - A: numpy.ndarray with shape (1, m) containing activated output.
                m: number of examples
        """
        m = Y.shape[1]

        dz = A - Y
        dw = np.matmul(X, dz.T) / m
        db = np.sum(dz) / m

        self.__W -= (alpha * dw).T
        self.__b -= alpha * db
