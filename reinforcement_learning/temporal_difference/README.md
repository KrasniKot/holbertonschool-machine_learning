# Temporal Difference

## Tasks:

### 0. Monte Carlo:
Write the function ``def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):`` that performs the Monte Carlo algorithm:

- ``env`` is environment instance
- ``V`` is a ``numpy.ndarray`` of shape ``(s,)`` containing the value estimate
- ``policy`` is a function that takes in a state and returns the next action to take
- ``episodes`` is the total number of episodes to train over
- ``max_steps`` is the maximum number of steps per episode
- ``alpha`` is the learning rate
- ``gamma`` is the discount rate
- Returns: ``V``, the updated value estimate

### 1. TD(λ):
Write the function ``def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):`` that performs the ``TD(λ)`` algorithm:

- ``env`` is the environment instance
- ``V`` is a ``numpy.ndarray`` of shape ``(s,)`` containing the value estimate
- ``policy`` is a function that takes in a state and returns the next action to take
- ``lambtha`` is the eligibility trace factor
- ``episodes`` is the total number of episodes to train over
- ``max_steps`` is the maximum number of steps per episode
- ``alpha`` is the learning rate
- ``gamma`` is the discount rate
- Returns: ``V``, the updated value estimate

### 2. SARSA(λ):
