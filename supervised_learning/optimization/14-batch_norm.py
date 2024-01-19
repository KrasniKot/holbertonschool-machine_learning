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
    dl = tf.keras.layers.Dense(
            units=n,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                mode='fan_avg'))

    mean, variance = tf.nn.moments(dl(prev), axes=[1])

    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))

    x_batch_norm = tf.nn.batch_normalization(
            x, mean, variance, beta, gamma, 1e-8)

    return activation(x_batch_norm)
