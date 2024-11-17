# Policy Gradients

## Tasks:

### 0. Simple Policy function:
Write a function ``def policy(matrix, weight):`` that computes the policy with a weight of a matrix.

### 1. Compute the Monte-Carlo policy gradient:
By using the previous function created ``policy``, write a function def ``policy_gradient(state, weight):`` that computes the Monte-Carlo policy gradient based on a state and a weight matrix.

- ``state``: matrix representing the current observation of the environment
- ``weight``: matrix of random weight
- Return: the action and the gradient (in this order)

### 2. Implement the training:
Write a function ``def train(env, nb_episodes, alpha=0.000045, gamma=0.98):`` that implements a full training.

- ``env``: initial environment
- ``nb_episodes``: number of episodes used for training
- ``alpha``: the learning rate
- ``gamma``: the discount factor
- You should use ``policy_gradient = __import__('policy_gradient').policy_gradient``
- Return: all values of the score (sum of all rewards during one episode loop)
- You need print the current episode number and the score after each loop in a format: ``Episode: {} Score: {}``

###
