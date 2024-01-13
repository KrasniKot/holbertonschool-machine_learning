#!/usr/bin/env python3
""" This module contains forward_prop which returns the output of the nn """

import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ Performs the forward propagation for a neural network
       - x: placeholder for the input data,
       - layer_sizes: nodes number list for each layer.
       - activations: list of activation functions for each layer.
    """
    prev = x

    for sz, act in zip(layer_sizes, activations):
        prev = create_layer(prev, sz, act)

    return prev
