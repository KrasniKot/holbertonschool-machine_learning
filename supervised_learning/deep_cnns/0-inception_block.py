#!/usr/bin/env python3
""" This module contains inception_block(),
    that builds an inception block as described in
    "Going Deeper with Convolutions (2014)":

    requires:
        - tensorflow.

"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """ Builds an inception block
        - A_prev: Previous layer output,
        - filters: tuple, contains:
            - F1: number of filters in the 1x1 convolution;
            - F3R: number of filters in the 1x1 convolution;
            - F3: number of filters in the 3x3 convolution;
            - F5R: number of filters in the 1x1 convolution;
            - F5: number of filters in the 5x5 convolution;
            - FPP: number of filters in the 1x1 convolution.
    """
    c = K.layers.Conv2D

    # Setting parameters
    pms = {"padding": "same", "activation": K.activations.relu,
           "kernel_initializer": K.initializers.he_normal()}

    # Building layers and getting their outputs
    # 1 * 1 Convolution for input
    l0 = c(filters=filters[0], kernel_size=(1, 1), **pms)(A_prev)

    # 1 * 1 Convolution for input
    l1 = c(filters=filters[1], kernel_size=(1, 1), **pms)(A_prev)

    # 3 * 3 Convolution for layer 1
    l2 = c(filters=filters[2], kernel_size=(3, 3), **pms)(l1)

    # 1 * 1 Convolution for input
    l3 = c(filters=filters[3], kernel_size=(1, 1), **pms)(A_prev)

    # 5 * 5 Convolution for convolution for layer 3
    l4 = c(filters=filters[4], kernel_size=(5, 5), **pms)(l3)

    # Max pooling layer for input
    mp = K.layers.MaxPooling2D(pool_size=(3, 3),
                               strides=(1, 1), padding='same')(A_prev)

    # 1 * 1 Connvolution for max pooling layer
    l5 = c(filters=filters[5], kernel_size=(1, 1), **pms)(mp)

    # Concatenating outputs
    return K.layers.concatenate([l0, l1, l2, l3, l4, mp, l5])
