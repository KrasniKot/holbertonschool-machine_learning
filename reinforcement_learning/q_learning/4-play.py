#!/usr/bin/env python3
""" Play an episode """

import numpy as np


def play(env, Q, max_steps=100):
    """ Play an episode using the trained Q-table """
    # 1. Reset the environment to get the initial state
    state = env.reset()[0]

    # 2. For each step in the episode
    total_rewards = 0
    rendered_outputs = []
    for _ in range(max_steps):
        # a. Render the current state of the environment
        rendered_outputs.append(env.render())

        # b.Choose action with the highest Q-value for the current state
        action = np.argmax(Q[state])

        # c. Take the action in the environment
        next_state, reward, done = env.step(action)[:3]

        # d. Update reward and update the next step
        total_rewards += reward
        state = next_state

        # e. Check episode has ended
        if done:
            break

    # 3. Display the final state of the environment
    rendered_outputs.append(env.render())

    return total_rewards, rendered_outputs
