#!/usr/bin/env python3
""" This module contains dropout_create_layer(),
    that creates a layer of a neural network using dropout.

    requires:
        - numpy,
        - tensorflow.
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ Creates a layer of a NN using dropout:

     - prev: Tensor containing the output of the previous layer
     - n: Number of nodes the new layer should contain
     - activation: Activation function that should be used on the layer
     - keep_prob: Probability that a node will be kept
    """
    kernel_ini = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    kernel_reg = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=kernel_ini,
                            kernel_regularizer=kernel_reg)

    return layer(prev)
