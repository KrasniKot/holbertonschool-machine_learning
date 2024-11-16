# Policy Gradients

## Tasks:

### 0. Simple Policy function:
Write a function ``def policy(matrix, weight):`` that computes the policy with a weight of a matrix.

### 1. Compute the Monte-Carlo policy gradient:
By using the previous function created ``policy``, write a function def ``policy_gradient(state, weight):`` that computes the Monte-Carlo policy gradient based on a state and a weight matrix.

- ``state``: matrix representing the current observation of the environment
- ``weight``: matrix of random weight
- Return: the action and the gradient (in this order)

###
###