#!/usr/bin/env python3
""" Perform Monte Carlo algorithm """

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """ Performs Monte Carlo algorithm
        - env .......... environment instance
        - V ............ numpy.ndarray of shape (s,) containing the value estimate
        - policy ....... function that takes in a state and returns the next action to take
        - episodes ..... total number of episodes to train over
        - max_steps .... maximum number of steps per episode
        - alpha ........ learning rate
        - gamma ........ discount rate

        Returns: The updated value estimate
    """
    for episode in range(episodes):  # Loop over the episodes
        edata = []

        state = env.reset()[0]  # 1. Start a new episode

        for _ in range(max_steps):
            ns, r, term, trunc, _ = env.step(policy(state))  # 2. Perform an action based on the policy
            edata.append((state, r))  # 3. Store the state and reward

            if term or trunc:
                break

            state = ns

        ######## Backward calculation of returns
        edata = np.array(edata, dtype=int)
        G     = 0  # Initialize the return
        for state, reward in edata[::-1]:
            G = reward + gamma * G                            # Discounted return
            if state not in edata[:episode, 0]:               # Check state is not already present in the episode data
                V[state] = V[state] + alpha * (G - V[state])  # Update value estimate
        ########

    return V
