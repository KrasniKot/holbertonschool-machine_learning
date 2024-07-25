#!/usr/bin/env python3
""" This module contains mean_cov(),
    that returns the mean and the covariance
    for the given data
"""

import numpy as np


def mean_cov(X):
    """ Calculates the mean and the covariance for the given dataset
        - X: numpy.ndarray of shape (n, d) containing the data set
            - n: number of data points
            - d: number of dimensions on each datapoint
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, _ = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    cov = 1 / (n - 1) * np.matmul(X.T - mean.T, X - mean)

    return mean, cov
