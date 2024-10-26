#!/usr/bin/env python3
""" Perform Q-learning """

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """ Performs Q-Learning
        - env ................ FrozenLakeEnv instance
        - Q .................. numpy.ndarray containing the Q-table
        - episodes  .......... total number of episodes to train over
        - max_steps .......... maximum number of steps per episode
        - alpha .............. learning rate
        - gamma .............. discount rate
        - epsilon ............ initial threshold for epsilon greedy
        - min_epsilon ........ minimum value that epsilon should decay to
        - epsilon_decay ...... decay rate for updating epsilon between episodes

        Returns:
        > updated Q-table
        > list containing the rewards per episode
    """
    # 1. Store total rewards for each episode
    trpe = []

    for episode in range(episodes):
        # 2. Set initial value of the state
        state = env.reset()  # Reset the environment for a new episode
        # print(state)
        state = state[0]

        total_reward = 0

        # 3. For each step in the episode:
        for step in range(max_steps):
            # a. Choose next action using epsilon-greedy
            naction = epsilon_greedy(Q, state, epsilon)

            # b. Perfom action
            nstate, reward, done = env.step(naction)[:3]

            # c. Check if episode has ended and the reward is 0
            if done and reward == 0:
                reward = -1

            # ####### d. Update the Q-value using the Q-learning formula:
            #    a[r + γ * max(Q(s' ,a')) - Q(s, a)]
            td = reward + gamma * np.max(Q[nstate]) - Q[state, naction]
            Q[state, naction] += alpha * td
            # #######

            total_reward += reward
            state = nstate

            if done:
                break  # End the episode if done

        trpe.append(total_reward)

        # 4. Update epsilon, ensure it doesn't drop below the minimum threshold
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q, trpe
