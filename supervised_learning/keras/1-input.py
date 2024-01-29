#!/usr/bin/env python3
""" This module contains build_model(),
    that builds a neural network with Keras.

    requires:
        - tensorflow.
        - numpy.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds a model using Keras.
        - nx: number of input features to the network,
        - layers: list containing the number of nodes for each layer,
        - activations: list containing the activation functions for each layer,
        - lambtha: L2 regularization parameter,
        - keep_prob: probability that a node will be kept for dropout.
    """
    inpt = K.Input(shape=(nx,))  # input tensor
    l2 = K.regularizers.L2(lambtha)
    n = len(layers)

    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=l2)(inpt)  # input layer out. tensor

    for i in range(n):
        if i != 0:
            x = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=l2)(x)

        if i < n - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    return K.Model(inputs=inpt, outputs=x)
