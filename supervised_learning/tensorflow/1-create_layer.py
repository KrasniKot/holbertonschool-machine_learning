#!/usr/bin/env python3
""" This module conains create_layer()
    which creates a layer
"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """ Creates a layer
         - prev: tensor output of the previous layer
         - n: number of nodes in the layer to create
         - activation: activation function that the layer should use
    """
    i = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=i, name='layer')

    return layer(prev)
