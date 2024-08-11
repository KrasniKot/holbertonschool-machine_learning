#!/usr/bin/env python3
""" Performs the Viterbi algorithm """

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ Performs the Viterbi algorithm for a hidden markov model
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
        N = Emission.shape[0]

        # Initialize the viterbi and backpointer matrices
        viterbi = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)

        # Initialization step
        viterbi[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
        backpointer[:, 0] = 0

        # Recursion step
        for t in range(1, T):
            for s in range(N):
                mx = np.max
                trans_prob = viterbi[:, t - 1] * Transition[:, s]
                viterbi[s, t] = mx(trans_prob) * Emission[s, Observation[t]]
                backpointer[s, t] = np.argmax(trans_prob)

        # Termination step
        P = np.max(viterbi[:, T - 1])
        best_last_state = np.argmax(viterbi[:, T - 1])

        # Backtracking to find the best path
        path = [best_last_state]
        for t in range(T - 1, 0, -1):
            best_last_state = backpointer[best_last_state, t]
            path.insert(0, best_last_state)

        return path, P

    except Exception as e:
        return None, None
