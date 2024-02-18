#!/usr/bin/en python3
""" This module contains identity_block(),
    that builds an identity block as described in
    "Deep Residual Learning for Image Recognition (2015)"

    requires:
        - tensorflow.
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ Builds an identity block:
        - A_prev: output from the previous layer,
        - filters: tuple, contains:
            - F11: number of filters in the first (1, 1) convolution,
            - F3: number of filters in the (3, 3) convolution,
            - f12: number of filters in the second (1, 1) convolution,
    """
    # Extract number of filter for each layer
    L0, L1, L2 = filters

    # Set common parameters and an activation function alias
    pms = {"padding": "same", "kernel_initializer": K.initializers.he_normal()}
    relu = K.activations.relu

    # Getting first (1, 1) convolutional layer output, normalized and activated
    L0 = K.layers.Conv2D(filters=L0, kernel_size=(1, 1), **pms)(A_prev)
    anL0 = K.layers.Activation(relu)(K.layers.BatchNormalization(axis=3)(L0))

    # Getting sec (3, 3) convolutional layer output, normalized and activated
    L1 = K.layers.Conv2D(filters=L1, kernel_size=(3, 3), **pms)(anL0)
    anL1 = K.layers.Activation(relu)(K.layers.BatchNormalization(axis=3)(L1))

    # Getting third (1, 1) convolutional layer output, normalized and activated
    L2 = K.layers.Conv2D(filters=L2, kernel_size=(1, 1), **pms)(anL1)
    nL2 = K.layers.BatchNormalization(axis=3)(L2)

    # Returning the activated addition
    return K.layers.Activation(relu)(K.layers.Add()([nL2, A_prev]))
