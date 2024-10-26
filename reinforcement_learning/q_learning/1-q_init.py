#!/usr/bin/env python3
""" Initialize the q-table """

import numpy as np


def q_init(env):
    """ Initializes the q-table
        - env ...... environment

        > Returns: The q-table
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
