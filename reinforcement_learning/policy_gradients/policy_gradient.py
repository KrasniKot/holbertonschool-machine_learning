#!/usr/bin/env python3
""" Have a function perform Monte Carlo Policy Gradient """

import numpy as np


def policy(matrix, weight):
    """ Computes the policy of a matrix
        - matrix ..... 2D array representing a row vector of features for a single state
        - weight ..... 2D matrix  with shape (state features, actions).
                       Each row corresponds to the weights applied to each feature.
    """
    z  = matrix @ weight         # Linear combination
    ez = np.exp(z - np.max(z))  # Stability improvement

    return ez / np.sum(ez)


def policy_gradient(state, weight):
    """ Computes the Monte-Carlo policy gradient based on a state and a weight matrix
        - state ...... matrix representing the current observation of the environment
        - weight ..... matrix of random weight
    """

    π = policy(state, weight)          # 1. Compute the action probabilities
    a = np.random.choice(len(π), p=π)  # 2. Sample an action using the action probabilities

    ######## 3. Compute the gradient of the log-probability of the chosen action
    one_hot    = np.zeros_like(π)  # 3a. Create a vector of zeros with the same shape as π
    one_hot[a] = 1                 # 3b. Set the element corresponding to the chosen action a to 1.
    ########

    return a, np.outer(state, one_hot - π)
