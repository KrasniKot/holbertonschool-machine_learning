#!/usr/bin/env python3
""" Defines function that performs (PCA) on a given dataset
"""

import numpy as np


def pca(X, ndim):
    """
    Performs principal components analysis (PCA) on a dataset

        - X: dataset
            n: number of data points
            d: number of dimensions in each data point
        - ndim: new dimensionality of the transformed X
    """
    mean = np.mean(X, axis=0, keepdims=True)

    A = X - mean
    u, s, v = np.linalg.svd(A)

    return np.matmul(A, v.T[:, :ndim])
