#!/usr/bin/env python3
""" Function that loads the pre-made
    environment FrozenLakeEnv from gymnasium
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """ Loads the pre-made FrozenLakeEnv environment from gymnasium
        - desc ........ None or a list of lists containing a custom description
                        of the map to load for the environment
        - map_name .... None or a string containing the pre-made map to load
        - is_slippery . boolean to determine if the ice is slippery

        > Returns: The environment
    """
    return gym.make("FrozenLake-v1",
                    desc=desc,
                    map_name=map_name,
                    is_slippery=is_slippery)
