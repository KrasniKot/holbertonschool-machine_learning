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


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
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
    fepsilon = epsilon

    for episode in range(episodes):    # 1. For each episode
        es     = np.zeros_like(Q)                   #   1a. Set e(s) to zero
        state  = env.reset()[0]                     #   1b. Reset the environment to get current state
        action = epsilon_greedy(Q, state, epsilon)  #   1c. Determine first action using epsilon greedy

        for _ in range(max_steps):  # 2. For each step
            ns, r, term, trunc, _ = env.step(action)                           # 2a. Take the action

            naction           = epsilon_greedy(Q, ns, epsilon)                 # 2b. Choose next action
            δ                 = r + gamma * Q[ns, naction] - Q[state, action]  # 2c. Compute TD error
            es[state, action] += 1                                             # 2d. Update e(s, a) adding 1
            es                *= lambtha * gamma                               # 2e. Decay the eligibility traces
            Q                 += alpha * δ * es                                # 2f. Update Q-values using eligibility traces

            if term or trunc:                                                  # 2g. Check episode still lives
                break

            state  = ns
            action = naction

        epsilon = min_epsilon + (fepsilon - min_epsilon) * np.exp(-epsilon_decay * episode)  # 3. Decay epsilon at the end of each episode

    return Q
