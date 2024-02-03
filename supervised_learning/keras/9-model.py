#!/usr/bin/env python3
""" This module contains functions that handle saving and loading a model.

    functions:
        - save_model(): saves a model;
        - load_model(): loads a model.

    requires:
        - tensorflow.
"""

import tensorflow.keras as K


def save_model(network, filename):
    """ Saves a model.
        - network: model to be saved,
        - filename: file path where the model should be saved to.
    """
    network.save(filename)


def load_model(filename):
    """ Loads a model.
        - filename: file path where the model should be loaded from.
    """
    return k.saving.load_model(filepath, compile=True)
