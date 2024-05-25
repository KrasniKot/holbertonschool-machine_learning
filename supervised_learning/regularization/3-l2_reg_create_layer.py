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
    l2_reg = tf.contrib.layers.l2_regularizer(lambtha)
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    layer = tf.layers.Dense(
        n,
        activation=activation,
        name="layer",
        kernel_initializer=weights_initializer,
        kernel_regularizer=l2_reg)
    return (layer(prev))
