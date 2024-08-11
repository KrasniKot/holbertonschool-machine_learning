#!/usr/bin/env python3
""" Performs the backward step in the forward-backward algorithm """

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ Performs the forward algorithm for a hidden markov model
        - Observartion: numpy.ndarray of shape (T,),
                        contains the index of the observation
            - T: number of observations
        - Emission: numpy.ndarray of shape (N, M), contains the emission
                    probability of a specific observation given a hidden state
            - N: number of hidden states
            - M: number of all possible observations
        - Transition: numpy.ndarray of shape (N, N),
                      contains the transition probabilities
        - Initial: numpy.ndarray of shape (N, 1), contains the probability
                   of starting in a particular hidden state
    """
    try:
        T = Observation.shape[0]
        N, M = Emission.shape
        B = np.empty([N, T], dtype='float')

        # 1. Initialization
        # There are no observations left to account for
        # after the final observation
        B[:, T - 1] = 1

        # 2. Regression
        tr = Transition
        for t in reversed(range(T - 1)):
            B[:, t] = tr @ (Emission[:, Observation[t + 1]] * B[:, t + 1])

        # 3. Termination
        return (Initial.T @ (Emission[:, Observation[0]] * B[:, 0])).item(), B

    except Exception:
        return None, None
