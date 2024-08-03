#!/usr/bin/env python3
""" initializes the variables for a GMM """

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ initializes variables for a Gaussian Mixture Model
        - X: is a numpy.ndarray of shape (n, d) containing the data set
        - k: is a positive integer containing the number of clusters
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None

    _, d = X.shape

    # Covariance matrix initialization
    S = np.zeros((k, d, d))
    S[:] = np.identity(d)

    # Mixture weigths initialization
    pi = np.zeros((k))
    pi[:] = 1 / k

    return pi, kmeans(X, k)[0], S
