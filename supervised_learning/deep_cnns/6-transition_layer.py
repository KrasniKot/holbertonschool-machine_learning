#!/usr/bin/env python3
""" This module contains transition_layer(),
    that builds a transition layer as described in:
    "Densely Connected Convolutional Networks"

    requires:
        - tensorflow.
"""

from tensorflow import keras as K


def transition_layer(X, nf, compression):
    """ Builds a transition layer:
        - X: previus layer's output,
        - nb_filters: integer, represents the number of filters in X,
        - grouwth_rate: growth rate for the dense block,
        - layers: is the number of layers in the dense block.
    """
    relu = K.activations.relu  # relu alias

    # Previous layer's normalized activated output
    anLX = K.layers.Activation(relu)(K.layers.BatchNormalization(axis=3)(X))

    nb_filters *= compression  # recalculating number of filters
    nf = int(nf)

    # First convolution (3, 3) layer's output
    L0 = K.layers.Conv2D(filters=nf, kernel_size=(1, 1), padding='same',
                         kernel_initializer=K.initializers.he_normal())(anLX)

    # Last average pooling layer's output
    AP = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                   padding="valid")(AP)

    return AP, nf
