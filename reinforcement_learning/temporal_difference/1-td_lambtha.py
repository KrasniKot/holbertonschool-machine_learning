#!/usr/bin/env python3
""" Have a function perform TD(λ) algorithm """

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """ Performs TD(λ) algorithm
        - env .........  environment instance
        - V ........... numpy.ndarray of shape (s,) containing the
        - policy ...... function that takes in a state and returns
        - lambtha  .... eligibility trace factor
        - episodes .... total number of episodes to train over
        - max_steps ... maximum number of steps per episode
        - alpha ....... learning rate
        - gamma ....... discount rate
    """
    for episode in range(episodes):
        es = np.zeros_like(V)
        state = env.reset()[0]

        for step in range(max_steps):
            ns, r, term, trunc, _ = env.step(policy(state))

            δ = r + gamma * V[ns] - V[state]
            es *= lambtha * gamma
            es[state] += 1
            V = V + alpha * δ * es

            if term or trunc:
                break

            state = ns

    return V
