#!/usr/bin/env python3
""" This module computes the Principal Components Analysis for a dataset """

import numpy as np


def pca(X, var=0.95):
    """ Performs principal components analysis (PCA) on a dataset

        - X: dataset
            n: number of data points
            d: number of dimensions in each data point
        - var: fraction of the variance that the PCA
               transformation should maintain
    """
    u, s, v = np.linalg.svd(X)

    rs = list(x / np.sum(s) for x in s)
    variance = np.cumsum(rs)
    nd = np.argwhere(variance >= var)[0, 0]

    return v.T[:, :(nd + 1)]
