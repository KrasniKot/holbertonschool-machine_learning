#!/usr/bin/env python3
""" This module l2_reg_create_layer(),
    that creates a layer that includes L2 regularization

    requires:
        - numpy,
        - tensorflow.
"""

import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ Creates a layer that includes L2 regularization.
        - prev: tensor containing the output of the previous layer,
        - n: number of nodes the new layer should contain,
        - activation: activation function that should be used on the layer,
        - lambtha: L2 regularization parameter.
    """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)

    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=reg)

    return layer(prev)
