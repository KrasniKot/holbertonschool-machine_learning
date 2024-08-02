#!/usr/bin/env python3
""" This module computes the intra-cluster variance """

import numpy as np


def variance(X, C):
    """ Calculates the total intra-cluster variance for a data set
        - X: is a numpy.ndarray of shape (n, d) containing the data set
        - C: is a numpy.ndarray of shape (k, d) containing the centroid means for
             each cluster

    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # Calculate the squared distances from each point to each centroid
    distances_squared = np.sum((X - C[:, np.newaxis])**2, axis=2)

    # Find the minimum squared distance for each point
    min_distances_squared = np.min(distances_squared, axis=0)

    # Sum the minimum squared distances to get the total intra-cluster variance
    return np.sum(min_distances_squared)  
