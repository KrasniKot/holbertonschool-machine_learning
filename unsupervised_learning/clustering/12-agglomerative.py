#!/usr/bin/env python3
""" Performs agglomerative clustering """

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """ Performs agglomerative clustering on a dataset
        - X: is a numpy.ndarray of shape (n, d) containing the dataset
        - dist: is the maximum cophenetic distance for all clusters
    """
    Z = scipy.cluster.hierarchy.linkage(X, method="ward")

    dendrogram = scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    plt.show()

    return clss