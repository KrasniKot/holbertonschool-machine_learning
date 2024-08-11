#!/usr/bin/env python3
""" Performs the fowrward step in the forward-backward algorithm """

import numpy as np


def forward(Observation, Emission, Transition, Initial):
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
        F = np.zeros([N, T], dtype='float')

        # 1. Initialization
        # Probability of being in each state at time t = 1 and observing O1
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        # 2. induction (Recursion)
        for t in range(1, T):
            F[:, t] = F[:, t - 1] @ Transition.T * Emission[:, Observation[t]]

        # 3. Termination
        return np.sum(F[:, T - 1]), F

    except Exception:
        return None, None
