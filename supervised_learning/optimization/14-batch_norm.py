#!/usr/bin/env python3
""" This module contains create_batch_norm_layer(),
    that creates a batch normalization layer for a nn.

    requires:
        - tensorflow,
        - numpy.
"""

import tensorflow as tf


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

    x = dl(prev)
    mean, variance = tf.nn.moments(x, axes=[0])

    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))

    nbatch = tf.nn.batch_normalization(
            x, mean, variance, beta, gamma, 1e-8)

    return activation(nbatch)
