#!/usr/bin/env python3
""" Calculates GMM from a dataset """

import sklearn.mixture


def gmm(X, k):
    """ Calculates a GMM from a dataset
        - X: is a numpy.ndarray of shape (n, d) containing the dataset
        - k: is the number of clusters
    """
    Gmm = sklearn.mixture.GaussianMixture(k)

    params = Gmm.fit(X)
    clss = Gmm.predict(X)

    return (params.weights_, params.means_,
            params.covariances_, clss, Gmm.bic(X))