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
        - lambda: L2 regularization parameter.
    """
    i = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")
    a = tf.layers.Dense(
        n,
        activation,
        kernel_initializer=i,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(lambtha))

    return a(prev)
