#!/usr/bin/env python3
""" Computes the p(t) """

import numpy as np


def markov_chain(P, s, t=1):
    """ Determines the probability of a markov chain being in a
        particular state after a specified number of iterations:
        - P: square 2D numpy.ndarray of shape (n, n),
            represents the transition matrix
             - n is the number of states in the markov chain
        - s: a numpy.ndarray of shape (1, n) representing the
             probability of starting in each state
        - t: the number of iterations that the markov chain has been through
    """

    # Compute the state after t steps by p(0) * P**t
    return np.dot(s, np.linalg.matrix_power(P, t))
