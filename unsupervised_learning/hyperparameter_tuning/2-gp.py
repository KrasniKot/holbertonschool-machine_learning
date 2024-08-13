#!/usr/bin/env python3
""" Class GaussianProcess that represents a noiseless 1D Gaussian process """

import numpy as np


class GaussianProcess:
    """ Defines a GaussianProcess """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ Initializes a GaussianProcess
            - X_init: numpy.ndarray of shape (t, 1); the inputs already sampled
                      with the black-box function
            - Y_init: numpy.ndarray of shape (t, 1); outputs already sampled
                      with the black-box function
                > l: number of initial states
            - l: length parameter for the kernel
            - sigma_f: standard deviation given to the output
                       of the black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ Calculates the covariance between two matrices
            - X1: numpy.ndarray of shape (m, 1)
            - X2: numpy.ndarray of shape (n, 1)
        """
        m, n = X1.shape[0], X2.shape[0]
        K = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                num = (X1[i] - X2[j]) ** 2
                denom = 2 * self.l ** 2

                K[i][j] = self.sigma_f ** 2 * np.exp(-num / denom)

        return K

    def predict(self, X_s):
        """ Predicts the mean and std deviation of points in a Gaussian process
            - X_s: numpy.ndarray of shape (s, 1); all of the points whose mean
                and standard deviation should be calculated.
        """
        common = self.kernel(self.X, X_s).T @ np.linalg.inv(self.K)

        mu = common @ self.Y
        s2 = np.diag(self.kernel(X_s, X_s) - common @ self.kernel(self.X, X_s))

        return mu.flatten(), s2

    def update(self, X_new, Y_new):
        """ Updates the model given the new data
            - X_new: numpy.ndarray of shape (1,) that represents
                     the new sample point
            - Y_new: numpy.ndarray of shape (1,) that represents
                     the new sample function value
        """
        self.X = np.append(self.X, [X_new], axis=0)
        self.Y = np.append(self.Y, [Y_new], axis=0)

        self.K = self.kernel(self.X, self.X)
