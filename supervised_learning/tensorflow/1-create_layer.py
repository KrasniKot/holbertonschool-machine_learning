#!/usr/bin/env python3
""" This module conains create_layer()
    which creates a layer
"""

import import tensorflow as tf


def create_layer(prev, n, activation):
    """ Creates a layer
         - prev: tensor output of the previous layer
         - n: number of nodes in the layer to create
         - activation: activation function that the layer should use
    """
    i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layer.Dens(
            units=n,
            activation=activation,
            kernel_initializer=i,
            name='layer')

    return layer(prev)
