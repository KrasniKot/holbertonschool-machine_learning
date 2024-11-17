#!/usr/bin/env python3
""" Have a function optimize the policy """

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """ Optimize the policy using Monte Carlo Gradient Policy
        - env ............ Gymnasium CartPole-v1 environment
        - nb_episodes .... Number of episodes to train over
        - alpha .......... Learning rate
        - gamma .......... Discount factor
        - show_result .... If True, render environment each 1000 episodes
    """
    # Initlize weights with the shape of (state, actions)
    θ = np.random.rand(env.observation_space.shape[0], env.action_space.n)

    S = []  # Store the scores for every episode
    for episode in range(nb_episodes):
        s = env.reset()[0]      # Get initial state
        Rs = []                  # Store rewards
        egradients = []                  # Store the gradients for each episode

        done = False                     # Determine whether the episode ended
        while not done:
            # Calculate action probabilities and gradients
            action, gradient = policy_gradient(s, θ)

            # Take the action in the environment
            ns, r, term, trunc, _ = env.step(action)

            # Store the reward and gradient for a step
            Rs.append(r)
            egradients.append(gradient)

            # Perform next action and determine if the episode lives
            s = ns
            done = term or trunc

        # Render the environment every 1000 episodes if show_result is True
        if show_result and episode % 1000 == 0:
            env.render()

        # Calculate total reward for the episode and store it
        S.append(sum(Rs))

        print(f"Episode: {episode} Score: {sum(Rs)}")

        for t, gradient in enumerate(egradients):
            # Calculate the discounted cumulative return
            Gt = sum([R * gamma ** R for R in Rs[t:]])
            θ += alpha * gradient * Gt  # Update the weights

    return S
