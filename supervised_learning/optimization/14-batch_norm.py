#!/usr/bin/env python3
""" This module contains create_batch_norm_layer(),
    that creates a batch normalization layer for a nn.

    requires:
        - tensorflow,
        - numpy.
"""

import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """ Creates a batch normalization layer for a nn.
        - prev: activated output of the previous layer,
        - n: number of nodes in the layer to be created,
        - activation: activation function to be used,
            on the output of the layer.
    """
    i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        n,
        activation=activation,
        name="layer",
        kernal_initializer=i)

    x = layer[prev]
    gamma = tf.Variable(tf.constant(
        1, shape=(1, n), trainable=True, name="gamma"))
    beta = tf.Variable(tf.constant(
        0, shape=(1, n), trainable=True, name="gamma"))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-8)
