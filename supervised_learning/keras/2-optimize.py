#!/usr/bin/env python3
""" This module contains optimize_model(),
    that sets up Adam optimization for a keras model
    with categorical crossentropy loss and accuracy metrics.

    requires:
        - tensorflow.
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ Sets up Adam optimization for a keras model
        with categorical crossentropy loss and accuracy metrics.
        - network: model to optimize,
        - alpha: learning rate,
        - beta: first Adam optimization parameter,
        - beta2: second Adam optimization parameter.
    """
    o = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(loss='categorical_crossentropy', optimizer=o,
                    metrics=['accuracy'])
