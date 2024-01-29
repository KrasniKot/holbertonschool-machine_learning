#!/usr/bin/env python3
""" This module contains build_model(),
    that builds a neural network with the Keras library

    requires:
        - tensorflow.
        - numpy.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds a models using Keras.
        - nx: number of input features to the network,
        - layers: list containing the number of nodes for each layer,
        - activations: list containing the activation functions for each layer,
        - lambtha: L2 regularization parameter,
        - keep_prob: probability that a node will be kept for dropout.
    """
    model = K.Sequential()
    L2 = K.regularizers.l2(lambtha)
    n = len(layers)

    # Input layer
    model.add(K.layers.Dense(layers[0], activation=activations[0], kernel_regularizer=L2, input_shape=(nx,)))

    for i in range(1, n):
        # Hidden layers
        model.add(K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=L2))

        if i < n - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
