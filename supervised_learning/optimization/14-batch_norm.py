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
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init, name='layer')
    y = layer(prev)

    mean, var = tf.nn.moments(y, axes=[0])
    ga = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True, name='gamma')
    be = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True, name='beta')
    norma = tf.nn.batch_normalization(y, mean, var, be, ga, 1e-8)

    return activation(norma)
