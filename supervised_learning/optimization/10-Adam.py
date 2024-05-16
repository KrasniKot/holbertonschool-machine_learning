#!/usr/bin/env python3
""" This module contains create_Adam_op(),
    which trains a nn by Adam.

    requires:
        - matplotlib,
        - numpy,
        - tensorflow.
"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ Trains a nn using Adam.
        - loss: nn los function,
        - alpha: learning rate,
        - beta1: weight used for the first moment,
        - beta2: weight used for the sencond moment,
        - epsilon: small number to prevent division by 0.
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
