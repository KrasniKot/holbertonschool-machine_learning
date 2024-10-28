#!/usr/bin/env python3
""" Epsilon Greedy algorithm """

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ Determine the next action using epsilon-greedy
        - q .......... numpy array containing q table
        - state ...... current state
        - epsilon .... epsilon to use for the calculation

        > Returns: next action index
    """
    # Get p to determine whether the agent should explore or exploit
    # If p is less than epsilon then explore
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, Q.shape[1])
    # If p is greater or equals than epsilon, then exploit
    else:
        return np.argmax(Q[state])
