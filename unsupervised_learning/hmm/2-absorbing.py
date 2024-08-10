#!/usr/bin/env/python 3
""" Checks whether a Markov chain is absorbing """

import numpy as np


def absorbing(P):
    """ Determines if a markov chain is absorbing
        - P: standard transition matrix
    """
    # Number of states
    n = P.shape[0]

    absorbing_states = [i for i in range(n) if P[i, i] == 1]
    if not absorbing_states:
        return False

    # Matrix to check reachability to absorbing states
    reachable = np.zeros((n, n), dtype=bool)

    # If there is a non-zero probability of moving from i to j, [i, j] = True
    reachable[P > 0] = True

    # Reachability propagation
    for _ in range(n):
        reachable = reachable | (reachable @ reachable)

    # Can each state reach any absorbing state?
    for i in range(n):
        if not any(reachable[i, j] for j in absorbing_states):
            return False

    return True
