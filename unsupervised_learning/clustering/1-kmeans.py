#!/usr/bin/env python3
""" This module handles the clusters' centroids for K-means """

import numpy as np


def kmeans(X, k, iterations=1000):
    """ Performs K-means on a dataset
        - X: numpy.ndarray of shape (n, d) containing the dataset
            - n:  number of data points
            - d: number of dimensions for each data point
        - k: positive integer containing the number of clusters
        - iterations: positive integer containing the max number of iterations
    """

    C, low, high = initialize(X, k)

    for i in range(iterations):
        # find all the points to the nearest cluster centroid
        clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
        newCentroid = np.copy(C)

        # Recomputing position for every centroid, for every cluster
        for c in range(k):
            # If a cluster contains no data points during the update step
            if c not in clss:
                newCentroid[c] = np.random.uniform(low, high)
            # else recompute centroid with average of all points
            else:
                newCentroid[c] = np.mean(X[clss == c], axis=0)

        # if Centroids of newly formed clusters do not change return
        if np.array_equal(newCentroid, C):
            return (C, clss)
        # else assign new centroids to recompute it
        else:
            C = newCentroid

    clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)

    return (C, clss)


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
    # minimum and maximum values of X along each dimension in d
    low, high = np.min(X, axis=0), np.max(X, axis=0)


    # initialize cluster centroids with multivariate uniform distribution
    return np.random.uniform(low, high, size=(k, X.shape[1])), low, high


