#!/usr/bin/env python3
""" Performs the Bayessian Optimization algorithm """

import numpy as np
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
