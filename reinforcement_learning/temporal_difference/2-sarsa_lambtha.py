#!/usr/bin/env python3
""" Have a function perform SARSA(λ) """

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ Determine next action to take using epsilon-greedy
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


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """ Performs SARSA(λ)
        - env ............. environment instance
        - Q ............... numpy.ndarray of shape (s,a) containing the Q table
        - lambtha ......... eligibility trace factor
        - episodes ........ total number of episodes to train over
        - max_steps ....... maximum number of steps per episode
        - alpha ........... learning rate
        - gamma ........... discount rate
        - epsilon ......... initial threshold for epsilon greedy
        - min_epsilon ..... minimum value that epsilon should decay to
        - epsilon_decay ... decay rate for updating epsilon between episodes
    """
    for episode in range(episodes):
        es = np.zeros_like(Q)
        state = env.reset()[0]
        action = epsilon_greedy(Q, state, epsilon)

        for step in range(max_steps):
            ns, r, term, trunc, _ = env.step(action)

            naction = epsilon_greedy(Q, state, epsilon)
            δ = r + gamma * Q[ns, naction] - Q[state, action]
            es[state, action] += 1
            es *= lambtha * gamma
            Q += alpha * δ * es

            if term or trunc:
                break

            state = ns
            action = naction

        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay))

    return Q
