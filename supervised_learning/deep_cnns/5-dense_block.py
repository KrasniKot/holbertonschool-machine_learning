#!/usr/bin/env python3
""" This module contains dense_block(),
    that builds a dense block as defined in
    "Densely Connected Convolutional Networks":

    requires:
        - tensorflow
"""

from tensorflow import keras as K


def dense_block(X, nb_filters, gr, layers):
    """ Builds a dense block:
        - X: output from the previous layer.
        - nb_filters: number of filters in X.
        - gr: growth rate for the dense block,
        - layers: number of layers in dense block.
    """
    # Setting common parameters and activation function alias
    pms = {"padding": "same", "kernel_initializer": K.initializers.he_normal()}
    ru = K.activations.relu

    for lr in range(layers):
        # Activated normalized first layer's output
        anL0 = K.layers.Activation(ru)(K.layers.BatchNormalization(axis=3)(X))

        # Activated normalized bottleneck layer's output before the (3, 3) conv
        L1 = K.layers.Conv2D(filters=(4 * gr), kernel_size=(1, 1), **pms)(anL0)
        anL1 = K.layers.Activation(ru)(K.layers.BatchNormalization(axis=3)(L1))

        # Final (3, 3) layer's output
        L2 = K.layers.Conv2D(filters=gr, kernel_size=(3, 3), **pms)(anL1)

        X = K.layers.concatenate([X, L2])  # Concatenating input and the blocks output
        nb_filters += gr  # Recalculating filters number

    return X, nb_filters
