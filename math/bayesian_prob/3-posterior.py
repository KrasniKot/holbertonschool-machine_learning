#!/usr/bin/env python3
""" This module contains marginal() that calculates the marginal
    probability of obtaining the data
"""

import numpy as np


def posterior(x, n, P, Pr):
    """ Calculates the posterior probability for the various
        hypothetical probabilities of developing severe side effects
        given the data
        - x: total number of patients that develop severe side effects
        - n total number of patients observed
        - P: containing the various hypothetical probabilities
            of developing severe side effects
        - Pr: containing the prior beliefs of P
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x)is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for value in range(P.shape[0]):
        if P[value] > 1 or P[value] < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
        if Pr[value] > 1 or Pr[value] < 0:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if np.isclose([np.sum(Pr)], [1]) == [False]:
        raise ValueError("Pr must sum to 1")

    return intersection(x, n, P, Pr) / marginal(x, n, P, Pr)


def marginal(x, n, P, Pr):
    """ Calculates the marginal probability of obtaining the data
        - x total number of patients that develop severe side effects
        - n: total number of patients observed
        - P: containing the various hypothetical probabilities
             of developing severe side effects
        - Pr: containing the prior beliefs of P
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for value in range(P.shape[0]):
        if P[value] > 1 or P[value] < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
        if Pr[value] > 1 or Pr[value] < 0:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if np.isclose([np.sum(Pr)], [1]) == [False]:
        raise ValueError("Pr must sum to 1")

    return np.sum(intersection(x, n, P, Pr))


def intersection(x, n, P, Pr):
    """ Calculates the intersection of obtaining varios hypothetical
        probabilities
        - x: total number of patients that develop severe side effects
        - n: total number of patients observed
        - P: containing the various hypothetical probabilities
            of developing severe side effects
        - Pr: containing the prior beliefs of P
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for value in range(P.shape[0]):
        if P[value] > 1 or P[value] < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
        if Pr[value] > 1 or Pr[value] < 0:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if np.isclose([np.sum(Pr)], [1]) == [False]:
        raise ValueError("Pr must sum to 1")

    return likelihood(x, n, P) * Pr


def likelihood(x, n, P):
    """ Calculates the likelihood for a binomial distribution
        - x: number of patients that develop severe side effects
        - n: total number of patients observed
        - P: 1D numpy.ndarray containing the various hypothetical probabilities
             of developing severe side effects.
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for value in P:
        if value > 1 or value < 0:
            raise ValueError("All values in P must be in the range [0, 1]")

    f = np.math.factorial

    return f(n) / (f(n - x) * f(x)) * (P ** x) * ((1 - P) ** (n - x))
