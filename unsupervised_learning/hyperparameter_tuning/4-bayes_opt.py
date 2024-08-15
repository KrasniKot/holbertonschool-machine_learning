#!/usr/bin/env python3
""" Performs the Bayessian Optimization algorithm """

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """ Defines a Bayessian Optimization algorithm """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """ Initializes a Bayessian Optimization algorithm
            - f: black-box function to be optimized
            - X_init: numpy.ndarray of shape (t, 1) representing
                      the inputs already sampled with the black-box function
            - Y_init: numpy.ndarray of shape (t, 1) representing
                      the outputs of the black-box function
                      for each input in X_init
                - t: number of initial samples
            - bounds: tuple of (min, max) representing the bounds
                      of the space in which to look for the optimal point
            - ac_samples: number of samples that should be
                          analyzed during acquisition
            - l: is the length parameter for the kernel
            - sigma_f: standard deviation given to the output of
                       the black-box function
            - xsi: exploration-exploitation factor for acquisition
            - minimize: bool determining whether optimization should
                        be performed for minimization (True)
                        or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        # ac_samples evenly spaced values between min_bound and max_bound
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """ Calculates the next best sample location """
        # Predictions and stddev
        mu, sigma = self.gp.predict(self.X_s)

        with np.errstate(divide='warn'):
            if self.minimize:
                mu_sample = np.min(self.gp.Y)
                improvement = (mu_sample - mu - self.xsi)
            else:
                mu_sample = np.amax(self.gp.Y)
                improvement = (mu - mu_sample - self.xsi)

            Z = improvement / sigma
            EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0  # Handle division by zero if sigma is 0

        # Next best sample point
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
