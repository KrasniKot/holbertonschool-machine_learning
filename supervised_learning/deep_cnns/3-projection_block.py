#!/usr/bin/env python3
""" This module contains projection_block(),
    that builds a projection block as described in
    "Deep Residual Learning for Image Recognition (2015)":

    requires:
        - tensorflow.
"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """ Builds a projection block:
        - A_prev: output from the previous layer,
        - filters: tuple or list, contains:
            - F11: number of filters in the first (1, 1) convolution,
            - F3: number of filters in the second (3, 3) convolution,
            - F12: number of filters in the third (1, 1) convolution.
        - s: stride of the first convolution.
    """
    # Extracting number of filters for each layer
    F11, F3, F12 = filters

    # Setting common parameters and activation function alias
    pms = {"padding": "same", "kernel_initializer": K.initializers.he_normal()}
    relu = K.activations.relu

    # Getting the activated normalized first layer output
    L0 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=s,
                         **pms)(A_prev)
    anL0 = K.layers.Activation(relu)(K.layers.BatchNormalization(axis=3)(L0))

    # Getting the activated normalized second layer output
    L1 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), **pms)(anL0)
    anL1 = K.layers.Activation(relu)(K.layers.BatchNormalization(axis=3)(L1))

    # Getting the normalized third layer output
    L2 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), **pms)(anL1)
    nL2 = K.layers.BatchNormalization(axis=3)(L2)

    # Getting the normalized shortcut layer output
    shct = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=s,
                           **pms)(A_prev)
    nshct = K.layers.BatchNormalization(axis=3)(shct)

    # Returning the addition of the third and the shortcut layer
    return K.layers.Activation(relu)(K.layers.Add()([nL2, nshct]))
