#!/usr/bin/env python3
""" This module contains create_RMSProp_op(),
    which performs the training for a nn using RMSProp.

    requires:
        - numpy,
        - tensorflow,
        - matplotlib.
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ Performs the training operation for a nn by RMSProp.
        - loss: loss function of the network,
        - alpha: learning rate,
        - beta2: RMSProp weight,
        - epsilon: small number to prevent division by zero.
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
