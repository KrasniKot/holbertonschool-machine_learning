#!/usr/bin/env python3
""" Train an agent that can play Atari's Breakout """

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent



def __set_up_env(envname="ALE/Breakout-v5"):
    """ Sets up the environment where the model will perform """
    # 1. Initialize the environment as rgb array so it can be fed to the CNN
    env = gym.make(envname, render_mode="rgb_array")

    # 2. Preprocess the environment’s observations
    env = AtariPreprocessing(env, screen_size=90, grayscale_obs=True,
                             frame_skip=1,
                             noop_max=30)

    return env


def __build_agent_cnn():
    """ Builds a CNN to process the game frames """
    model = Sequential()

    return model


def __set_up_DQN_agent(ws, nba, cnnmodel, ti):
    """ Sets up the DQN agent
        - ws ......... determines how many frames make up a state
        - nba ........ number of possible actions to perform
        - cnnmodel ... cnn model to process game frames
        - ti ......... how frequently the to update the CNN
    """
    # 1. Set memory limit and policy to use
    memory = SequentialMemory(limit=1000000, window_length=ws)
    policy = EpsGreedyQPolicy()

    # 2. Build the DQN agent
    # PROCESOR?
    dqn = DQNAgent(model=cnnmodel,
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=50000,
                   target_model_update=10000,
                   policy=policy,
                   enable_dueling_network=True
                   delta_clip=1
                   train_interval=ti
                   gamma=.99)

    # 3. Compile DQN agent
    dqn.compile(optimizer='adam', metrics=['mae'])

    return dqn


def __plot_tprogress():
    """ Plots the training progress """


def __save_tmodel():
    """ Saves the trained model """


if __name__ == "__main__":
    # 1. Set up the environment
    env      = __set_up_env()        # TO DO

    # 2. Build CNN model for the agent
    cnnmodel = __build_agent_cnn()   # TO DO

    # 3. Set up the DQN agent
    dqnagent = __set_up_DQN_agent()  # TO DO

    # 4. Train the agent
    history  = dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)

    # 5. Plot progress
    __plot_tprogress()               # TO DO

    # 6. Save the trained model
    __save_tmodel()                  # TO DO
