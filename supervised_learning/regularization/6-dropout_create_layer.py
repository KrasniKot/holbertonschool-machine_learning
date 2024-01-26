#!/usr/bin/env python3
""" This module contains dropout_create_layer(),
    that creates a layer of a neural network using dropout.

    requires:
        - numpy,
        - tensorflow.
"""

import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ Creates a layer of a neural network using dropout.
        - prev: tensor containing the output of the previous layer,
        - n: number of nodes the new layer should contain,
        - activation: activation function that should be used on the layer,
        - keep_prob: probability that a node will be kept.
    """
    i = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))

    a = tf.layers.Dense(
        n,
        activation,
        kernel_initializer=i,
        kernel_regularizer=tf.layers.Dropout(keep_prob))

    return a(prev)
