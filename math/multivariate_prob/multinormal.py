#!/usr/bin/env python3
""" This module handles multivariate gausian distributions """

import numpy as np


class MultiNormal:
    """ Represents a Multivariate Gaussian Distribution """

    def __init__(self, data):
        """ Initializes a MultiNormal instance
            - data: represents the number of dimensions and datapoints
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.matmul(data - self.mean, data.T - self.mean.T) / (n - 1)

    def pdf(self, x):
        """ Calculates the PDF at a given data point
            - x: numpy.ndarray of shape (d, 1), contains the data point whose
                 PDF should be calculated
                - d: number of dimensions of the instance
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")

        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        test_d, one = x.shape

        if test_d != d or one != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        n = self.mean.shape[0]
        diff = x - self.mean
        inv_Sigma = np.linalg.inv(self.cov)
        det_Sigma = np.linalg.det(self.cov)
        norm_const = 1.0 / (np.power((2 * np.pi), n / 2) * np.sqrt(det_Sigma))
        exponent = -0.5 * np.dot(np.dot(diff.T, inv_Sigma), diff)

        return (norm_const * np.exp(exponent)).item()
