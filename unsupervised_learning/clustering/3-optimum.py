#!/usr/bin/env python3
""" Finds the optimum number of clusters by variance """

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ Tests for the optimum number of clusters by variance
        - X: is a numpy.ndarray of shape (n, d) containing the data set
        - kmin: is a positive integer containing the minimum number of clusters
                to check for (inclusive)
        - kmax: is a positive integer containing the maximum number of clusters
                to check for (inclusive)
        - iterations: is a positive integer containing the maximum number of
                      iterations for K-means
    """
    if type(X) is not np.ndarray:
        return (None, None)

    if type(kmin) is not int:
        return (None, None)

    if kmax is not None and type(kmax) is not int:
        return (None, None)
    if kmax is None:
        kmax = X.shape[0]

    if len(X.shape) != 2 or kmin < 1:
        return (None, None)

    if kmax is not None and kmax <= kmin:
        return (None, None)

    if type(iterations) is not int:
        return (None, None)
    if iterations <= 0:
        return (None, None)

    results = []
    varsdiff = []
    for k in range(kmin, kmax + 1):
        cluster, clss = kmeans(X, k, iterations)
        results.append((cluster, clss))
        varsdiff.append(variance(X, results[0][0]) - variance(X, cluster))

    return results, varsdiff
