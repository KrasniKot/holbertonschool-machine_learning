#!usr/bin/env/python3
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

        self.__W = np.random.normal(size=(nx, 1))
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
