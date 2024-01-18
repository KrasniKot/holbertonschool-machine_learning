#!/usr/bin/env python3
""" This module contains create_Adam_op(),
    which trains a nn by Adam.

    requires:
        - matplotlib,
        - numpy,
        - tensorflow.
"""

import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ Trains a nn using Adam.
        - loss: nn los function,
        - alpha: learning rate,
        - beta1: weight used for the first moment,
        - beta2: weight used for the sencond moment,
        - epsilon: small number to prevent division by 0.
    """
    return tf.train.AdamOptimizer(
            alpha, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss)
