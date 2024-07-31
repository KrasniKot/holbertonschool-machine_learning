#!/usr/bin/env python3
""" This module handles the clusters' centroids for K-means """

import numpy as np


def initialize(X, k):
    """ Initializes the clusters' centroids for K-means
        - X: numpy.ndarray of shape (n, d), contains the dataset
             that will be used for K-means clustering
             - n: number of data points
             - d: number of dimensions for each data point
        -k:  positive integer containing the number of clusters
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    # minimum values of X along each dimension in d
    low = np.min(X, axis=0)
    # maximum values of X along each dimension in d
    high = np.max(X, axis=0)

    # initialize cluster centroids with multivariate uniform distribution
    centroids = np.random.uniform(low, high, size=(k, X.shape[1]))
    return centroids
