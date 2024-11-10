#!/usr/bin/env python3
""" Have a function perform TD(λ) algorithm """

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """ Performs TD(λ) algorithm
        - env .........  environment instance
        - V ........... numpy.ndarray of shape (s,) containing the value estimate
        - policy ...... function that takes in a state and returns the next action to take
        - lambtha  .... eligibility trace factor
        - episodes .... total number of episodes to train over
        - max_steps ... maximum number of steps per episode
        - alpha ....... learning rate
        - gamma ....... discount rate
    """
    for episode in range(episodes):  # 1. For each episode
        es    = np.zeros_like(V)     #   1a. Set e(s) to zero
        state = env.reset()[0]       #   1b. Reset the environment to get current state

        for step in range(max_steps):                        # 2. Simulate the episode
            ns, r, term, trunc, _ = env.step(policy(state))  #   2a. Perform an action

            δ          = r + gamma * V[ns] - V[state]        #   2c. Compute TD
            es        *= lambtha * gamma                     #   2d. Update the eligibility for each state
            es[state] += 1
            V          = V + alpha * δ * es                  #   2e. Update the value estimates for all states

            if term or trunc:                                #   2b. Check the episode still lives 
                break

            state = ns

    return V
