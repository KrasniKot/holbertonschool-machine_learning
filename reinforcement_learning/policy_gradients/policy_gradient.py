#!/usr/bin/env python3
""" Have a function perform Monte Carlo Policy Gradient """

import numpy as np


def policy(matrix, weight):
    """ Computes the policy of a matrix
        - matrix ..... 2D array representing a row vector of features
        - weight ..... 2D matrix  with shape (state features, actions)
                       Each row corresponds to the weights applied to
    """
    z = matrix @ weight         # Linear combination
    ez = np.exp(z - np.max(z))  # Stability improvement

    return ez / np.sum(ez)


def policy_gradient(state, weight):
    """ Computes the Monte-Carlo policy gradient based on a stat
        - state ...... matrix representing the current observati
        - weight ..... matrix of random weight
    """

    π = policy(state, weight)          # 1. Compute the action pro
    a = np.random.choice(len(π), p=π)  # 2. Sample an action usin

    # ####### 3. Compute the gradient of the log-probability of the ch
    one_hot = np.zeros_like(π)  # 3a. Create a vector of zeros w
    one_hot[a] = 1                 # 3b.
    # #######

    return a, np.outer(state, one_hot - π)
