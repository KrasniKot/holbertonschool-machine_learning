#!/usr/bin/env python3
""" This module conains functions that handle
    saving and loading a model's weights.

    contains:
        - save_weights: saves model's weights;
        - load_weights: loads model's weights.

    requires:
        - tensorflow.
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ Saves a model's weights.
        - network: model whose weights should be saved,
        - filename: file path where the weights shoud be saved to,
        - save_format: format in which the weights should be saved.
    """
    network.save_weights(filepath, overwrite=True)


def load_weights(network, filename):
    """ Loads a model's weights.
        - network: model whose weights should be loaded,
        - filename: file path where the weights shoud be loaded from.
    """
    network.load_weights(filepath)
