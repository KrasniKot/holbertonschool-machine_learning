#!/usr/bin/env python3
""" This module handles model's configuration saving and loading:

    contians:
        - save_config: saves model's config;
        - load_config: loads model's config.

    requires:
        - tensorflow.
"""

import tensorflow.keras as K


def save_config(network, filename):
    """ Saves a model's configuration:
        - network: model whose configuration should be saved,
        - filename: file path where the configuration shoud be saved to.
    """
    n = network.to_json()
    with open(filename, "w") as f:
        f.write(n)


def load_config(filename):
    """ Loads a model's configuration:
        - filename: file path where the configuration shoud be loaded from.
    """
    f = open(filename, 'r')
    m = f.read()
    f.close()

    return K.models.model_from_json(m)
