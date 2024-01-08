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

    def cost(self, Y, A):
        """ Calculates the cost of the model
                - Y: numpy.ndarray with shape (1, m)
                    containing the correct output.
                - A: A is a numpy.ndarray with shape (1, m)
                    containing the activated output.
                    - m: number of examples.
        """

        m = Y.shape[1]
        return (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1.0000001-A))

    def evaluate(self, X, Y):
        """ Evaluates cost and A of the model
                - X: numpy.ndarray with shape (nx, m) containing the input features.
                - Y: numpy.ndarray with shape (1, m) containing the correct output.
        """
        self.forward_prop(X)
        c = self.cost(Y, self.__A2)
        a = np.where(self.__A2 >= 0.5, 1, 0)

        return a, c

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """  Calculates one pass of gradient descent on the Neural Network
                - X: numpy.ndarray with shape (nx, m) containing the input features.
                - Y: numpy.ndarray with shape (1, m) containing the correct output.
                - A1: output of the hidden layer.
                - A2: predicted output.
                - alpha: learning rate.
        """

        dZ2 = A2 - Y
        dW2 = (dZ2 @ A1.T) / X.shape[1]
        db2 = np.sum(dZ2, axis=1, keepdims=True) / X.shape[1]

        dZ1 = np.dot(self.W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (dZ1 @ X.T) / X.shape[1]
        db1 = np.sum(dZ1, axis=1, keepdims=True) / X.shape[1]

        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
