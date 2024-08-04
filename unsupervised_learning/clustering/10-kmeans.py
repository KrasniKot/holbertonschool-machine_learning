#!/usr/bin/env python3
""" Performs K-means algorithm using sklearn """

import sklearn.cluster


def kmeans(X, k):
    """ Performs K-means on a dataset using sklearn
        - X: is a numpy.ndarray of shape (n, d) containing the dataset
        - k: is the number of clusters
    """
    k_model = sklearn.cluster.KMeans(n_clusters=k).fit(X)

    return k_model.cluster_centers_, k_model.labels_
