#!/usr/bin/env python3
""" Performs the Baum Welch algorithm """

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """ Performs the Baum-Welch algorithm for a hidden markov
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
        iterations: is the number of times expectation-maximization should be
                    performed
    """
    try:
        T = Observations.shape[0]
        M, N = Emission.shape

        for n in range(1, iterations):
            alpha = forward(Observations, Emission, Transition, Initial)
            beta = backward(Observations, Emission, Transition, Initial)
            xi = np.zeros((M, M, T - 1))
            for i in range(T - 1):
                denominator = np.dot(np.dot(alpha[:, i].T, Transition) *
                                     Emission[:, Observations[i + 1]].T,
                                     beta[:, i + 1])
                for j in range(M):
                    numerator = alpha[j, i] * Transition[j] *\
                        Emission[:, Observations[i + 1]].T * beta[:, i + 1].T
                    xi[j, :, i] = numerator / denominator
            gamma = np.sum(xi, axis=1)
            gam = gamma
            Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
            gam = np.hstack((gam,
                             np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
            denominator = np.sum(gamma, axis=1)
            for i in range(N):
                Emission[:, i] = np.sum(gamma[:, Observations == i], axis=1)
            Emission = np.divide(Emission, denominator.reshape((-1, 1)))
        return (Transition, Emission)
    except Exception:
        return None, None


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
    T = Observation.shape[0]
    N, M = Emission.shape
    B = np.empty([N, T], dtype='float')

    B[:, T - 1] = 1

    tr = Transition
    for t in reversed(range(T - 1)):
        B[:, t] = tr @ (Emission[:, Observation[t + 1]] * B[:, t + 1])

    return B


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
    T = Observation.shape[0]
    N, M = Emission.shape
    F = np.zeros([N, T], dtype='float')

    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        F[:, t] = Transition.T @ F[:, t - 1] * Emission[:, Observation[t]]

    return F
