#!/usr/bin/env python3
""" Have a trained agent play Breakout """

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

import numpy as np

import pygame
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Permute
from tensorflow.keras.models import load_model

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.callbacks import Callback


class PygameCallback(Callback):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.running = True

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Breakout Game DQN Play")

    def on_action_end(self, action, logs=None):
        """Called at the end of each action taken by the agent."""
        if not self.running:
            return

        # Render the environment
        frame = self.env.render(mode='rgb_array')  # Get the frame
        frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))  # Convert to Pygame surface
        frame = pygame.transform.scale(frame, (640, 480))
        self.screen.blit(frame, (0, 0))  # Draw the frame on the screen
        pygame.display.flip()  # Update the display

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()

    def close(self):
        """Clean up resources."""
        pygame.quit()


class EnvProc(Processor):
    """ Custom processor class for the environment """

    def process_observation(self, observation):
        """ Converts the observation to numpy array

        - observation ...... environment observation

        Returns: The numpy observation
        """
        return np.array(observation)

    def process_state_batch(self, batch):
        """ Normaalizes the pixel values ofa batch of states

        - batch ......... batch of states

        Returns: Normalized batch of states
        """
        return np.array(batch, dtype=np.float32) / 255.0

    def process_reward(self, reward):
        """ Clip the reward to be within the range [-1,1]

        - reward ......... Environment's reward

        Returns: Clipped reward
        """
        return np.clip(reward, -1, 1)


class CWrapper(gym.Wrapper):
    """ A compatibility wrapper to make Gymnasium
        environments compatible with keras-rl
    """
    def __init__(self, env):
        super().__init__(env)
        self.metadata['render_fps'] = 60  # Set to your desired FPS

    def step(self, action):
        """ Take a step in the env using the given action

        - action ........ action to be taken in the environment

        Returns:
        - observation .... obs from env after action taken
        - reward ......... reward obtain after taking the action
        - done ........... bool indicating whether episode has ended
        - info ........... additional information from the environment
        """
        obs, r, term, truncated, info = self.env.step(action)

        # Return in keras-rl expected format
        return obs, r, term or truncated, info

    def reset(self, **kwargs):
        """ Resets the evironment and returns the observation

        - kwargs ...... any additional needed parameters

        Returns:
        - initial obs of the env
        """
        # Keras-RL expects only the observation
        return self.env.reset(**kwargs)[0]

    def render(self, mode='human', **kwargs):
        """ Render the environment with the specified mode """
        # Remove `mode` if it causes an error
        if 'mode' in kwargs:
            try:
                # Try rendering with mode
                return self.env.render(mode=mode, **kwargs)
            except TypeError:
                # Retry rendering without `mode`
                kwargs.pop('mode')
                return self.env.render(**kwargs)
        else:
            # Render without `mode` if it wasn't provided
            return self.env.render(**kwargs)


def __set_up_env(envname="ALE/Breakout-v5"):
    """ Sets up the environment where the model will perform """
    # 1. Initialize the environment as rgb array so it can be fed to the CNN
    env = gym.make(envname, render_mode="rgb_array")

    # 2. Preprocess the environment’s observations
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True,
                             frame_skip=1, noop_max=30)

    # 3. Compatibility wrapping to ensure compatibility with keras rl
    return CWrapper(env)


def __build_agent_cnn(iS, nactions):
    """ Builds a CNN to process the game frames
        - iS ......... input shape for the model
        - nactinos ... number of possible actions for the model to perform

        Returns:
        - model
    """
    model = Sequential()

    # 1. Convolutional Layers to capture spatial features
    model.add(Permute((2, 3, 1), input_shape=iS))
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=iS))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))

    # 2. Flatten the output to feed into Dense layers
    model.add(Flatten())

    # 3. Fully connected (Dense) layers
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nactions, activation='linear'))  # Output layer for actions

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

    # 2. Build the DQN agent
    dqn = DQNAgent(model=cnnmodel,
                   nb_actions=nba,
                   policy=EpsGreedyQPolicy(),
                   memory=memory,
                   processor=EnvProc(),
                   nb_steps_warmup=50000,
                   target_model_update=10000,
                   delta_clip=1,
                   train_interval=ti,
                   gamma=.99)

    # 3. Compile DQN agent
    dqn.compile(optimizer='adam', metrics=['mae'])

    return dqn


if __name__ == '__main__':
    # 1. Create environment
    env = __set_up_env()

    # 2. Build CNN model for the agent
    # Input shape (4 window size, 84 . 84 screen dims)
    cnnmodel = __build_agent_cnn((4, 84, 84), env.action_space.n)
    cnnmodel.load_weights('./policy.h5')

    # 3. Set up the DQN agent
    dqnagent = __set_up_DQN_agent(4, env.action_space.n, cnnmodel, 4)

    # 4. Have the DNQ agent play breakout
    pc = PygameCallback(env)
    scores = dqnagent.test(env, nb_episodes=5, visualize=False, callbacks=[pc])
