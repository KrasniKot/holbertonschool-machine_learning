# Reinforcement Learning

## Tasks:

### 0. Load the Environment:
Write a function ``def load_frozen_lake(desc=None, map_name=None, is_slippery=False):`` that loads the pre-made ``FrozenLakeEnv`` evnironment from ``gymnasium``:

- ````desc```` is either ``None`` or a list of lists containing a custom description of the map to load for the environment
- map_name is either ``None`` or a string containing the pre-made map to load
- Note: If both ``desc`` and ``map_name`` are ``None``, the environment will load a randomly generated 8x8 map
- ``is_slippery`` is a boolean to determine if the ice is slippery
- Returns: the environment

### 1. Initialize Q-table:
Write a function ``def q_init(env):`` that initializes the Q-table:

- ``env`` is the ``FrozenLakeEnv`` instance
- Returns: the`` Q-table`` as a ``numpy.ndarray`` of zeros

### 2. Epsilon Greedy:
### 3. Q-learning:
### 4. Play: