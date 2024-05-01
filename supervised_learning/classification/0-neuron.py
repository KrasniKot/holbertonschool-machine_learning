#!/usr/bin/env python3
"""
This module contains the class Neuron
which performs binary classification

requires:
    - numpy
"""

import numpy as np


class Neuron():
    """ Defines a Neuron """
    def __init__(self, nx):
        """ Initializes a neuron
            - nx: number of input features to the neuron
        """
        if nx is int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
